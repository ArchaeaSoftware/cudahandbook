/*
 * scanDecoupledLookback2.cuh
 *
 * ILP-optimized decoupled look-back: each thread scans IPT contiguous elements
 * in registers (raising work per thread and enabling coalesced, vectorizable
 * I/O), a warp-shuffle block scan stitches the per-thread sums together, and
 * the same warp-cooperative tile look-back runs on top. Tile = blockDim.x*IPT.
 *
 * Reuses the descriptor from scanDecoupledLookback.cuh (include it first).
 */
#ifndef __SCAN_DECOUPLED_LOOKBACK2_CUH__
#define __SCAN_DECOUPLED_LOOKBACK2_CUH__

#include <cstdint>

#include "scanDecoupledLookback.cuh"

template<class T, int IPT>
__global__ void
scanDecoupledLookback2_kernel(
    T *out, const T *in, volatile scanStatus *status, uint32_t *tileCounter, size_t N )
{
    const int B = blockDim.x;
    extern __shared__ T s[];                 // B*IPT elements, logical order
    __shared__ uint32_t s_tile;
    __shared__ T s_base;
    __shared__ T warpsum[32];

    if ( threadIdx.x == 0 ) s_tile = atomicAdd( tileCounter, 1 );
    __syncthreads();
    const uint32_t tile = s_tile;
    const size_t tileBase = (size_t) tile * B * IPT;

    // Coalesced striped load; s[k] holds logical element k of the tile.
    #pragma unroll
    for ( int i = 0; i < IPT; i++ ) {
        int k = i * B + threadIdx.x;
        size_t g = tileBase + k;
        s[k] = ( g < N ) ? in[g] : (T) 0;
    }
    __syncthreads();

    // Per-thread sequential inclusive scan of a contiguous chunk (in registers).
    T chunk[IPT];
    const int c0 = threadIdx.x * IPT;
    T run = (T) 0;
    #pragma unroll
    for ( int i = 0; i < IPT; i++ ) { run += s[c0 + i]; chunk[i] = run; }
    const T threadSum = run;

    // Block exclusive scan of threadSum via warp shuffles -> offset, aggregate.
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, numWarps = B >> 5;
    T x = threadSum;
    #pragma unroll
    for ( int off = 1; off < 32; off <<= 1 ) {
        T y = __shfl_up_sync( 0xffffffffu, x, off );
        if ( lane >= off ) x += y;
    }
    if ( lane == 31 ) warpsum[wid] = x;
    __syncthreads();
    if ( wid == 0 ) {
        T w = ( lane < numWarps ) ? warpsum[lane] : (T) 0;
        #pragma unroll
        for ( int off = 1; off < 32; off <<= 1 ) {
            T y = __shfl_up_sync( 0xffffffffu, w, off );
            if ( lane >= off ) w += y;
        }
        if ( lane < numWarps ) warpsum[lane] = w;
    }
    __syncthreads();
    const T warpOffset = ( wid == 0 ) ? (T) 0 : warpsum[wid - 1];
    const T offset = ( x - threadSum ) + warpOffset;   // block-exclusive prefix
    const T aggregate = warpsum[numWarps - 1];

    scanCoopLookback<T>( status, tile, aggregate, s_base );
    __syncthreads();

    // Add tile prefix + block offset, store coalesced.
    const T base = s_base + offset;
    #pragma unroll
    for ( int i = 0; i < IPT; i++ ) s[c0 + i] = chunk[i] + base;
    __syncthreads();
    #pragma unroll
    for ( int i = 0; i < IPT; i++ ) {
        int k = i * B + threadIdx.x;
        size_t g = tileBase + k;
        if ( g < N ) out[g] = s[k];
    }
}

template<class T, int IPT>
void
scanDecoupledLookback2( T *out, const T *in, size_t N, int b )
{
    cudaError_t status_cudart;
    scanStatus *gStatus = 0; uint32_t *ctr = 0;

    if ( N == 0 ) return;
    size_t tile = (size_t) b * IPT;
    uint32_t numTiles = (uint32_t) ( ( N + tile - 1 ) / tile );
    // Transient scratch: allocate/zero stream-ordered on the default stream so
    // repeated calls recycle pool memory instead of synchronizing per malloc.
    cuda(MallocAsync( &gStatus, numTiles * sizeof(scanStatus), 0 ) );
    cuda(MemsetAsync( gStatus, 0, numTiles * sizeof(scanStatus), 0 ) );
    cuda(MallocAsync( &ctr, sizeof(uint32_t), 0 ) );
    cuda(MemsetAsync( ctr, 0, sizeof(uint32_t), 0 ) );
    scanDecoupledLookback2_kernel<T,IPT><<<numTiles, b, tile * sizeof(T)>>>(
        out, in, gStatus, ctr, N );
Error_cudart:
    if ( gStatus ) cudaFreeAsync( gStatus, 0 );
    if ( ctr )     cudaFreeAsync( ctr, 0 );
}

#endif // __SCAN_DECOUPLED_LOOKBACK2_CUH__
