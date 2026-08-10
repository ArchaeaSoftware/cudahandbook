/*
 *
 * sat.cuh
 *
 * Summed-area tables (integral images) for the normalized cross-correlation
 * denominator. Builds BOTH moment tables -- sum of pixels and sum of squares --
 * as unbordered, inclusive int64 SATs:  S(x,y) = sum over [0..x] x [0..y].
 * The four-corner rule (satBox) then gives any window's SumI/SumISq in O(1),
 * independent of template size and shared across a whole template bank or FFT
 * search.
 *
 * Four builds share one signature, selected by a template tag so they are
 * drop-in swappable:
 *   SAT_NAIVE -- one thread per row, then per column; both moments fused in one
 *                pass. Simple and illustrative; serial within a line.
 *   SAT_CUB   -- the library build: cub::DeviceScan row scans, a coalesced
 *                transpose, scan again (= columns), transpose back.
 *   SAT_DLB   -- CUB row scan into a uint32 intermediate (one row's running sum
 *                fits 32 bits out to W ~ 66K), then a transpose-free single-pass
 *                column scan using decoupled look-back (Chapter 13). No transpose,
 *                and the narrow intermediate halves both its DRAM traffic and its
 *                shared staging. ~3x the throughput of SAT_CUB at large sizes.
 *   SAT_FUSED -- the fast build: one kernel fuses the row scan INTO the column
 *                look-back, so the intermediate is never written or read. Each tile
 *                scans its rows with a horizontal look-back over the tiles to its
 *                left, then its columns with a vertical look-back over the tiles
 *                above; both moments ride the same chains and the image is read
 *                once. ~1.9x the throughput of SAT_DLB at large sizes.
 *
 * Scratch is caller-owned (query with satScratchBytes, allocate, pass in) so the
 * header holds no state and needs no destructor:
 *     size_t n = satScratchBytes<SAT_DLB>( W, H );
 *     void *scratch;  cudaMalloc( &scratch, n );
 *     satBuild<SAT_DLB>( dImg, W, H, dSum, dSumSq, scratch, n );
 *
 * Copyright (c) 2025-2026, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __SAT_CUH__
#define __SAT_CUH__

#include <cstdint>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <chError.h>

enum SATMethod { SAT_NAIVE, SAT_CUB, SAT_DLB, SAT_FUSED };   // illustrative / library / DLB / fused single-pass

// Four-corner box query over [x0,x1) x [y0,y1); S(-1,*) = S(*,-1) = 0.
__host__ __device__ __forceinline__ int64_t
satBox( const int64_t *S, int W, int x0, int y0, int x1, int y1 )
{
    int64_t d =                  S[(size_t)(y1-1)*W + (x1-1)];
    int64_t b = (y0 > 0)       ? S[(size_t)(y0-1)*W + (x1-1)] : 0;
    int64_t c = (x0 > 0)       ? S[(size_t)(y1-1)*W + (x0-1)] : 0;
    int64_t a = (x0>0 && y0>0) ? S[(size_t)(y0-1)*W + (x0-1)] : 0;
    return d - b - c + a;
}

namespace satdetail {

static const int SAT_BW = 64;   // DLB tile width (columns, multiple of 32)
static const int SAT_BR = 32;   // DLB stripe height

// Value iterators: widen a uint8 pixel (or its square) for the CUB row scan.
struct RowKey {
    int W;
    __host__ __device__ int64_t operator()( int64_t i ) const { return i / W; }
};
struct Pix64 {
    const uint8_t *p;
    __host__ __device__ int64_t operator()( int64_t i ) const { return (int64_t)p[i]; }
};
struct Pix64q {
    const uint8_t *p;
    __host__ __device__ int64_t operator()( int64_t i ) const { int64_t v = p[i]; return v*v; }
};
struct Pix32 {
    const uint8_t *p;
    __host__ __device__ uint32_t operator()( int64_t i ) const { return (uint32_t)p[i]; }
};
struct Pix32q {
    const uint8_t *p;
    __host__ __device__ uint32_t operator()( int64_t i ) const { uint32_t v = p[i]; return v*v; }
};

// ---- naive: one thread per row, then per column; both moments fused, in place ----

__global__ void
naiveRow( const uint8_t *img, int W, int H, int64_t *S, int64_t *Q )
{
    for ( int y = blockIdx.x*blockDim.x + threadIdx.x; y < H; y += gridDim.x*blockDim.x ) {
        int64_t a = 0;
        int64_t b = 0;
        for ( int x = 0; x < W; x++ ) {
            int v = img[(size_t)y*W + x];
            a += v;
            b += (int64_t)v*v;
            S[(size_t)y*W + x] = a;
            Q[(size_t)y*W + x] = b;
        }
    }
}

__global__ void
naiveCol( int W, int H, int64_t *S, int64_t *Q )
{
    for ( int x = blockIdx.x*blockDim.x + threadIdx.x; x < W; x += gridDim.x*blockDim.x ) {
        int64_t a = 0;
        int64_t b = 0;
        for ( int y = 0; y < H; y++ ) {
            a += S[(size_t)y*W + x];
            b += Q[(size_t)y*W + x];
            S[(size_t)y*W + x] = a;
            Q[(size_t)y*W + x] = b;
        }
    }
}

// ---- coalesced tiled transpose (inH x inW) -> (inW x inH), grid-strided ----

template<class T>
__global__ void
transposeT( const T *in, T *out, int inW, int inH )
{
    __shared__ T tile[32][33];
    for ( int by = blockIdx.y*32; by < inH; by += gridDim.y*32 ) {
        for ( int bx = blockIdx.x*32; bx < inW; bx += gridDim.x*32 ) {
            int x = bx + threadIdx.x;
            int y = by + threadIdx.y;
            if ( x < inW && y < inH ) {
                tile[threadIdx.y][threadIdx.x] = in[(size_t)y*inW + x];
            }
            __syncthreads();
            int ox = by + threadIdx.x;
            int oy = bx + threadIdx.y;
            if ( ox < inH && oy < inW ) {
                out[(size_t)oy*inH + ox] = tile[threadIdx.x][threadIdx.y];
            }
            __syncthreads();
        }
    }
}

// ---- single-pass decoupled-look-back column scan (IT intermediate -> int64 SAT) ----
//
// A block owns a (stripe, column-group) tile staged in shared, so A is read once.
// It publishes its per-column aggregate, spins on a status flag looking back over
// earlier stripes of the same columns for the running carry, then applies it.
// Tiles are handed out in stripe order by an atomic counter, so a block only ever
// waits on earlier-scheduled tiles -> forward progress. Launch to the resident limit.

template<class IT>
__global__ void
colDLB( const IT *__restrict__ A, int W, int H, int nS, int WG,
        int *ctr, volatile int *status, int64_t *aggr, int64_t *pref, int64_t *S )
{
    __shared__ IT  tile[SAT_BR][SAT_BW+1];
    __shared__ int s_tile;
    const int tid = threadIdx.x;
    const int numTiles = nS*WG;
    // Grab the first tile (stripe-major order via the atomic counter); the block
    // walks tiles until the counter passes the last -- the loop's exit criterion.
    if ( 0 == tid ) s_tile = atomicAdd( ctr, 1 );
    __syncthreads();
    for ( int t = s_tile; t < numTiles; t = s_tile ) {
        int s = t / WG;
        int g = t % WG;
        // col = this block's tid-th column; consecutive tids -> consecutive columns,
        // so the per-row global loads/stores below are coalesced across the warp.
        int col = g*SAT_BW + tid;
        int y0 = s*SAT_BR;
        int y1 = min( y0 + SAT_BR, H );
        bool ok = col < W;

        // stage the stripe, forming this tile's per-column aggregate
        int64_t agg = 0;
        for ( int r = 0; r < SAT_BR; r++ ) {
            int y = y0 + r;
            IT v = (ok && y < y1) ? A[(size_t)y*W + col] : (IT)0;
            tile[r][tid] = v;
            agg += (int64_t)v;
        }
        if ( ok ) aggr[(size_t)s*W + col] = agg;
        __syncthreads();
        __threadfence();
        if ( 0 == tid ) status[t] = 1;             // AGGREGATE ready

        // look back over earlier stripes for the exclusive prefix (the carry)
        int64_t ex = 0;
        for ( int p = s-1; p >= 0; p-- ) {
            // pi has no tid: status[pi] is block-uniform, so the whole block reads one
            // flag, spins together, and breaks together -- only ex below is per-column.
            int pi = p*WG + g;
            int st = status[pi];
            while ( 0 == st ) {                    // spin until predecessor ready
                st = status[pi];
            }
            __threadfence();
            if ( 2 == st ) {
                if ( ok ) ex += pref[(size_t)p*W + col];
                break;
            }
            if ( ok ) ex += aggr[(size_t)p*W + col];
        }
        if ( ok ) pref[(size_t)s*W + col] = ex + agg;
        __syncthreads();
        __threadfence();
        if ( 0 == tid ) status[t] = 2;             // PREFIX ready

        // apply the carry from shared and write the SAT
        int64_t a = ex;
        for ( int r = 0; r < SAT_BR; r++ ) {
            a += (int64_t)tile[r][tid];
            int y = y0 + r;
            if ( ok && y < y1 ) S[(size_t)y*W + col] = a;
        }
        __syncthreads();

        // grab the next tile; the for's next-step re-reads it as t = s_tile
        if ( 0 == tid ) s_tile = atomicAdd( ctr, 1 );
        __syncthreads();
    }
}

// ---- fully-fused single-pass 2D decoupled look-back (both moments, no intermediate) ----
//
// SAT_DLB still runs a CUB row scan into a uint32 intermediate before its column
// look-back. This kernel fuses the two directions so the intermediate is never
// written or read: a block stages one (SAT_BR x SAT_BW) tile in shared, scans its
// rows with a HORIZONTAL look-back over the tiles to its left, then scans its
// columns with a VERTICAL look-back over the tiles above. Both moments (sum and
// sum of squares) ride the same look-back chains, and the image is read once.
//
// Tiles are handed out in row-major order by the atomic counter, so both the left
// neighbor (t-1) and the upper neighbor (t-WG) are lower tile indices: a block only
// ever waits on earlier-scheduled tiles -> forward progress. Launch to the resident
// limit. Row carries fit uint32 (one row's running sum is <= W*255^2 < 2^32 out to
// W ~ 66K); column carries and the SAT itself are int64.

__global__ void
fusedBoth( const uint8_t *__restrict__ img, int W, int H, int nS, int WG,
           int *ctr, volatile int *statusH, volatile int *statusV,
           uint32_t *rowAggS, uint32_t *rowPrefS, uint32_t *rowAggQ, uint32_t *rowPrefQ,
           int64_t *colAggS, int64_t *colPrefS, int64_t *colAggQ, int64_t *colPrefQ,
           int64_t *S, int64_t *Q )
{
    __shared__ uint32_t tileS[SAT_BR][SAT_BW+1];   // pixels, then their row/column scan
    __shared__ uint32_t tileQ[SAT_BR][SAT_BW+1];   // pixel squares, likewise
    __shared__ uint32_t hcarryS[SAT_BR];           // horizontal (left) carry, per row
    __shared__ uint32_t hcarryQ[SAT_BR];
    __shared__ uint32_t rowLocS[SAT_BR];           // this tile's per-row totals
    __shared__ uint32_t rowLocQ[SAT_BR];
    __shared__ int      s_tile;
    const int tid = threadIdx.x;
    const int numTiles = nS*WG;
    // Grab the first tile (row-major order via the atomic counter); the block walks
    // tiles until the counter passes the last -- the loop's exit criterion.
    if ( 0 == tid ) s_tile = atomicAdd( ctr, 1 );
    __syncthreads();
    for ( int t = s_tile; t < numTiles; t = s_tile ) {
        int tj  = t / WG;               // stripe (row block)
        int ti  = t % WG;               // column group
        // col = this block's tid-th column; consecutive tids -> consecutive columns,
        // so the staging loads and the SAT stores below are coalesced across the warp.
        int col = ti*SAT_BW + tid;
        int y0  = tj*SAT_BR;
        int y1  = min( y0 + SAT_BR, H );
        bool ok = col < W;

        // 1. stage the stripe: read the image once, forming pixels and their squares
        for ( int r = 0; r < SAT_BR; r++ ) {
            int y = y0 + r;
            uint32_t p = (ok && y < y1) ? img[(size_t)y*W + col] : 0u;
            tileS[r][tid] = p;
            tileQ[r][tid] = p*p;
        }
        __syncthreads();

        // 2. per-row totals -- threads 0..SAT_BR-1 each cooperate as one row r --
        // then publish the row AGGREGATE for the horizontal look-back.
        if ( tid < SAT_BR ) {
            int r = tid;
            uint32_t s = 0;
            uint32_t q = 0;
            for ( int c = 0; c < SAT_BW; c++ ) {
                s += tileS[r][c];
                q += tileQ[r][c];
            }
            rowLocS[r] = s;
            rowLocQ[r] = q;
            rowAggS[(size_t)t*SAT_BR + r] = s;
            rowAggQ[(size_t)t*SAT_BR + r] = q;
        }
        __syncthreads();
        __threadfence();
        if ( 0 == tid ) statusH[t] = 1;            // row AGGREGATE ready

        // 3. horizontal look-back over the left tiles -> exclusive row carry.
        // statusH[pidx] carries no tid, so threads 0..SAT_BR-1 read one flag and spin
        // together; only the accumulated carry below is per row.
        if ( tid < SAT_BR ) {
            int r = tid;
            uint32_t exS = 0;
            uint32_t exQ = 0;
            for ( int p = ti-1; p >= 0; p-- ) {
                int pidx = tj*WG + p;
                int st = statusH[pidx];
                while ( 0 == st ) {                // spin until the left tile is ready
                    st = statusH[pidx];
                }
                __threadfence();
                if ( 2 == st ) {                   // its full prefix is done: take it, stop
                    exS += rowPrefS[(size_t)pidx*SAT_BR + r];
                    exQ += rowPrefQ[(size_t)pidx*SAT_BR + r];
                    break;
                }
                exS += rowAggS[(size_t)pidx*SAT_BR + r];
                exQ += rowAggQ[(size_t)pidx*SAT_BR + r];
            }
            hcarryS[r] = exS;
            hcarryQ[r] = exQ;
            rowPrefS[(size_t)t*SAT_BR + r] = exS + rowLocS[r];
            rowPrefQ[(size_t)t*SAT_BR + r] = exQ + rowLocQ[r];
        }
        __syncthreads();
        __threadfence();
        if ( 0 == tid ) statusH[t] = 2;            // row PREFIX ready

        // 4. horizontal scan: inclusive prefix along each row + its carry, in place.
        // tileS/tileQ[r][c] now hold R(col,row), the row-scanned value.
        if ( tid < SAT_BR ) {
            int r = tid;
            uint32_t accS = hcarryS[r];
            uint32_t accQ = hcarryQ[r];
            for ( int c = 0; c < SAT_BW; c++ ) {
                accS += tileS[r][c];
                tileS[r][c] = accS;
                accQ += tileQ[r][c];
                tileQ[r][c] = accQ;
            }
        }
        __syncthreads();

        // 5. per-column totals of R -- every thread cooperates as its own column col --
        // then publish the column AGGREGATE for the vertical look-back.
        int64_t colLocS = 0;
        int64_t colLocQ = 0;
        for ( int r = 0; r < SAT_BR; r++ ) {
            colLocS += (int64_t)tileS[r][tid];
            colLocQ += (int64_t)tileQ[r][tid];
        }
        colAggS[(size_t)t*SAT_BW + tid] = colLocS;
        colAggQ[(size_t)t*SAT_BW + tid] = colLocQ;
        __syncthreads();
        __threadfence();
        if ( 0 == tid ) statusV[t] = 1;            // column AGGREGATE ready

        // 6. vertical look-back over the upper tiles -> exclusive column carry.
        // statusV[pidx] carries no tid: the whole warp reads one flag and spins
        // together, and only the carry is per column.
        int64_t vS = 0;
        int64_t vQ = 0;
        for ( int p = tj-1; p >= 0; p-- ) {
            int pidx = p*WG + ti;
            int st = statusV[pidx];
            while ( 0 == st ) {                    // spin until the upper tile is ready
                st = statusV[pidx];
            }
            __threadfence();
            if ( 2 == st ) {
                vS += colPrefS[(size_t)pidx*SAT_BW + tid];
                vQ += colPrefQ[(size_t)pidx*SAT_BW + tid];
                break;
            }
            vS += colAggS[(size_t)pidx*SAT_BW + tid];
            vQ += colAggQ[(size_t)pidx*SAT_BW + tid];
        }
        colPrefS[(size_t)t*SAT_BW + tid] = vS + colLocS;
        colPrefQ[(size_t)t*SAT_BW + tid] = vQ + colLocQ;
        __syncthreads();
        __threadfence();
        if ( 0 == tid ) statusV[t] = 2;            // column PREFIX ready

        // 7. vertical scan: apply the column carry down the tile, write both SATs
        int64_t accS = vS;
        int64_t accQ = vQ;
        for ( int r = 0; r < SAT_BR; r++ ) {
            accS += (int64_t)tileS[r][tid];
            accQ += (int64_t)tileQ[r][tid];
            int y = y0 + r;
            if ( ok && y < y1 ) {
                S[(size_t)y*W + col] = accS;
                Q[(size_t)y*W + col] = accQ;
            }
        }
        __syncthreads();

        // grab the next tile; the for's next-step re-reads it as t = s_tile
        if ( 0 == tid ) s_tile = atomicAdd( ctr, 1 );
        __syncthreads();
    }
}

__host__ inline size_t
alignUp( size_t x, size_t a )
{
    return (x + a - 1) / a * a;
}

// Scratch layout within the caller's buffer; identical arithmetic drives both
// satScratchBytes (base = 0, read .total) and satBuild (base = scratch, use offsets).
struct Plan {
    size_t cubBytes = 0;
    size_t total = 0;
    size_t offCub = 0;
    size_t offA = 0;         // CUB row-scan intermediate
    size_t offAt = 0;        // CUB transpose scratch
    size_t offBt = 0;
    size_t offA32 = 0;       // DLB uint32 intermediate
    size_t offAggr = 0;      // DLB per-column aggregates
    size_t offPref = 0;      // DLB per-column prefixes
    size_t offStatus = 0;    // DLB per-tile status flags
    size_t offCtr = 0;       // DLB/FUSED atomic tile counter
    size_t offStatusH = 0;   // FUSED per-tile horizontal status flags
    size_t offStatusV = 0;   // FUSED per-tile vertical status flags
    size_t offRowAggS = 0;   // FUSED per-tile-row aggregates/prefixes (uint32, both moments)
    size_t offRowPrefS = 0;
    size_t offRowAggQ = 0;
    size_t offRowPrefQ = 0;
    size_t offColAggS = 0;   // FUSED per-tile-column aggregates/prefixes (int64, both moments)
    size_t offColPrefS = 0;
    size_t offColAggQ = 0;
    size_t offColPrefQ = 0;
    int nS = 0;
    int WG = 0;
};

template<SATMethod M>
__host__ inline Plan
plan( int W, int H )
{
    Plan p;
    size_t N = (size_t)W*H;
    const size_t AL = 256;
    p.nS = (H + SAT_BR - 1) / SAT_BR;
    p.WG = (W + SAT_BW - 1) / SAT_BW;
    thrust::counting_iterator<int64_t> cnt( 0 );
    if constexpr ( SAT_NAIVE == M ) {
        p.total = 0;
    }
    else if constexpr ( SAT_CUB == M ) {
        cub::DeviceScan::InclusiveSumByKey( nullptr, p.cubBytes,
            thrust::make_transform_iterator( cnt, RowKey{W} ),
            thrust::make_transform_iterator( cnt, Pix64{nullptr} ), (int64_t *)nullptr, (int)N );
        size_t o = 0;
        p.offCub = o;
        o = alignUp( o + p.cubBytes, AL );
        p.offA = o;
        o = alignUp( o + N*8, AL );
        p.offAt = o;
        o = alignUp( o + N*8, AL );
        p.offBt = o;
        o = alignUp( o + N*8, AL );
        p.total = o;
    }
    else if constexpr ( SAT_DLB == M ) {
        cub::DeviceScan::InclusiveSumByKey( nullptr, p.cubBytes,
            thrust::make_transform_iterator( cnt, RowKey{W} ),
            thrust::make_transform_iterator( cnt, Pix32{nullptr} ), (uint32_t *)nullptr, (int)N );
        size_t o = 0;
        p.offCub = o;
        o = alignUp( o + p.cubBytes, AL );
        p.offA32 = o;
        o = alignUp( o + N*4, AL );
        p.offAggr = o;
        o = alignUp( o + (size_t)p.nS*W*8, AL );
        p.offPref = o;
        o = alignUp( o + (size_t)p.nS*W*8, AL );
        p.offStatus = o;
        o = alignUp( o + (size_t)p.nS*p.WG*4, AL );
        p.offCtr = o;
        o = alignUp( o + 4, AL );
        p.total = o;
    }
    else {   // SAT_FUSED
        size_t nt = (size_t)p.nS * p.WG;
        size_t o = 0;
        p.offCtr = o;
        o = alignUp( o + 4, AL );
        p.offStatusH = o;
        o = alignUp( o + nt*4, AL );
        p.offStatusV = o;
        o = alignUp( o + nt*4, AL );
        p.offRowAggS = o;
        o = alignUp( o + nt*SAT_BR*4, AL );
        p.offRowPrefS = o;
        o = alignUp( o + nt*SAT_BR*4, AL );
        p.offRowAggQ = o;
        o = alignUp( o + nt*SAT_BR*4, AL );
        p.offRowPrefQ = o;
        o = alignUp( o + nt*SAT_BR*4, AL );
        p.offColAggS = o;
        o = alignUp( o + nt*SAT_BW*8, AL );
        p.offColPrefS = o;
        o = alignUp( o + nt*SAT_BW*8, AL );
        p.offColAggQ = o;
        o = alignUp( o + nt*SAT_BW*8, AL );
        p.offColPrefQ = o;
        o = alignUp( o + nt*SAT_BW*8, AL );
        p.total = o;
    }
    return p;
}

} // namespace satdetail

// Scratch bytes this method needs for a W x H image.
template<SATMethod M>
__host__ inline size_t
satScratchBytes( int W, int H )
{
    return satdetail::plan<M>( W, H ).total;
}

// Build both moment SATs (dSum, dSumSq); int64, unbordered inclusive.
template<SATMethod M>
__host__ cudaError_t
satBuild( const uint8_t *dImg, int W, int H,
          int64_t *dSum, int64_t *dSumSq,
          void *scratch, size_t scratchBytes )
{
    using namespace satdetail;
    cudaError_t status_cudart = cudaSuccess;
    size_t N = (size_t)W*H;
    thrust::counting_iterator<int64_t> cnt( 0 );
    auto rowsW = thrust::make_transform_iterator( cnt, RowKey{W} );

    if constexpr ( SAT_NAIVE == M ) {
        naiveRow<<<(H+255)/256,256>>>( dImg, W, H, dSum, dSumSq );
        naiveCol<<<(W+255)/256,256>>>( W, H, dSum, dSumSq );
    }
    else {
        Plan p = plan<M>( W, H );
        if ( scratchBytes < p.total ) return cudaErrorInvalidValue;
        char *base = (char *)scratch;

        if constexpr ( SAT_CUB == M ) {
            void    *cubTmp = base + p.offCub;
            int64_t *A  = (int64_t *)(base + p.offA);
            int64_t *At = (int64_t *)(base + p.offAt);
            int64_t *Bt = (int64_t *)(base + p.offBt);
            dim3 blk( 32, 32 );
            for ( int which = 0; which < 2; which++ ) {
                int64_t *out = which ? dSumSq : dSum;
                size_t cb = p.cubBytes;
                if ( 0 == which ) {
                    CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( cubTmp, cb, rowsW,
                        thrust::make_transform_iterator( cnt, Pix64{dImg} ), A, (int)N ) );
                }
                else {
                    CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( cubTmp, cb, rowsW,
                        thrust::make_transform_iterator( cnt, Pix64q{dImg} ), A, (int)N ) );
                }
                transposeT<int64_t><<<dim3((W+31)/32,(H+31)/32),blk>>>( A, At, W, H );
                cb = p.cubBytes;
                CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( cubTmp, cb,
                    thrust::make_transform_iterator( cnt, RowKey{H} ), At, Bt, (int)N ) );
                transposeT<int64_t><<<dim3((H+31)/32,(W+31)/32),blk>>>( Bt, out, H, W );
            }
        }
        else if constexpr ( SAT_DLB == M ) {
            void     *cubTmp = base + p.offCub;
            uint32_t *A32    = (uint32_t *)(base + p.offA32);
            int64_t  *aggr   = (int64_t  *)(base + p.offAggr);
            int64_t  *pref   = (int64_t  *)(base + p.offPref);
            int      *status = (int *)(base + p.offStatus);
            int      *ctr    = (int *)(base + p.offCtr);
            int numSM;
            int perSM;
            CUDART_CHECK( cudaDeviceGetAttribute( &numSM, cudaDevAttrMultiProcessorCount, 0 ) );
            CUDART_CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &perSM, colDLB<uint32_t>, SAT_BW, 0 ) );
            int blocks = numSM * perSM;
            for ( int which = 0; which < 2; which++ ) {
                int64_t *out = which ? dSumSq : dSum;
                size_t cb = p.cubBytes;
                if ( 0 == which ) {
                    CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( cubTmp, cb, rowsW,
                        thrust::make_transform_iterator( cnt, Pix32{dImg} ), A32, (int)N ) );
                }
                else {
                    CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( cubTmp, cb, rowsW,
                        thrust::make_transform_iterator( cnt, Pix32q{dImg} ), A32, (int)N ) );
                }
                CUDART_CHECK( cudaMemset( ctr, 0, sizeof(int) ) );
                CUDART_CHECK( cudaMemset( status, 0, (size_t)p.nS*p.WG*sizeof(int) ) );
                colDLB<uint32_t><<<blocks,SAT_BW>>>( A32, W, H, p.nS, p.WG, ctr, status, aggr, pref, out );
            }
        }
        else {   // SAT_FUSED
            int      *ctr      = (int *)(base + p.offCtr);
            int      *statusH  = (int *)(base + p.offStatusH);
            int      *statusV  = (int *)(base + p.offStatusV);
            uint32_t *rowAggS  = (uint32_t *)(base + p.offRowAggS);
            uint32_t *rowPrefS = (uint32_t *)(base + p.offRowPrefS);
            uint32_t *rowAggQ  = (uint32_t *)(base + p.offRowAggQ);
            uint32_t *rowPrefQ = (uint32_t *)(base + p.offRowPrefQ);
            int64_t  *colAggS  = (int64_t *)(base + p.offColAggS);
            int64_t  *colPrefS = (int64_t *)(base + p.offColPrefS);
            int64_t  *colAggQ  = (int64_t *)(base + p.offColAggQ);
            int64_t  *colPrefQ = (int64_t *)(base + p.offColPrefQ);
            int numSM;
            int perSM;
            CUDART_CHECK( cudaDeviceGetAttribute( &numSM, cudaDevAttrMultiProcessorCount, 0 ) );
            CUDART_CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &perSM, fusedBoth, SAT_BW, 0 ) );
            int blocks = numSM * perSM;
            size_t nt = (size_t)p.nS * p.WG;
            CUDART_CHECK( cudaMemset( ctr, 0, sizeof(int) ) );
            CUDART_CHECK( cudaMemset( statusH, 0, nt*sizeof(int) ) );
            CUDART_CHECK( cudaMemset( statusV, 0, nt*sizeof(int) ) );
            // one launch: both moments through a single fused pass, image read once
            fusedBoth<<<blocks,SAT_BW>>>( dImg, W, H, p.nS, p.WG, ctr, statusH, statusV,
                                          rowAggS, rowPrefS, rowAggQ, rowPrefQ,
                                          colAggS, colPrefS, colAggQ, colPrefQ, dSum, dSumSq );
        }
    }

Error_cudart:
    return status_cudart;
}

#endif // __SAT_CUH__
