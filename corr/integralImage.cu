/*
 *
 * integralImage.cu
 *
 * The summed-area table (integral image) behind the normalized
 * cross-correlation denominator.
 *
 * NCC needs, for every candidate window, the sum of the underlying image
 * pixels (and, built identically on v*v, the sum of their squares). A
 * summed-area table sat(x,y) = sum of every pixel above and to the left of
 * (x,y) turns each window sum into a four-corner difference
 *     box[x0,x1) x [y0,y1) = sat(x1-1,y1-1) - sat(x0-1,y1-1)
 *                          - sat(x1-1,y0-1) + sat(x0-1,y0-1),
 * O(1) per output pixel and independent of the template size -- and the one
 * table serves every template in a bank, or every shift in an FFT search.
 *
 * PRECISION. A box sum is a difference of large corner values, so the table
 * must not overflow or the subtraction is garbage. A WxH 8-bit sum reaches
 * W*H*255; the companion sum-of-squares reaches W*H*255*255, which passes
 * 2^31 by 512x512. The table is therefore int64_t, which makes every
 * four-corner query exact for any image that fits in memory.
 *
 * CONSTRUCTION. The table is separable: an inclusive prefix sum along each
 * row, then down each column. Two builds are compared:
 *   naive -- one thread per row, then one thread per column. Simple, but each
 *            line is scanned serially by a single (row-strided, uncoalesced)
 *            thread; it underuses the GPU and is the real cost of an
 *            otherwise-cheap FFT matcher.
 *   CUB   -- cub::DeviceScan::InclusiveSumByKey, keyed by row index, runs
 *            every row scan at once with single-pass decoupled look-back
 *            (Chapter 13); a coalesced transpose turns the column pass into
 *            another row scan. The library-first answer, and it wins by more
 *            as the image grows.
 *
 * Build: nvcc -O3 -arch=sm_86 -I ../chLib integralImage.cu -o integralImage
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

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <chError.h>

// ---- naive separable build: one thread per row, then one thread per column ----
__global__ void
serialRow( const uint8_t *img, int W, int H, int64_t *sat )
{
    for ( int y = blockIdx.x*blockDim.x + threadIdx.x; y < H; y += gridDim.x*blockDim.x ) {
        int64_t acc = 0;
        for ( int x = 0; x < W; x++ ) {
            acc += img[(size_t)y*W + x];
            sat[(size_t)y*W + x] = acc;
        }
    }
}

__global__ void
serialCol( int W, int H, int64_t *sat )
{
    for ( int x = blockIdx.x*blockDim.x + threadIdx.x; x < W; x += gridDim.x*blockDim.x ) {
        int64_t acc = 0;
        for ( int y = 0; y < H; y++ ) {
            acc += sat[(size_t)y*W + x];
            sat[(size_t)y*W + x] = acc;
        }
    }
}

// ---- coalesced tiled transpose of an (inH x inW) int64 array -> (inW x inH) ----
__global__ void
transpose( const int64_t *in, int64_t *out, int inW, int inH )
{
    __shared__ int64_t tile[32][33];       // +1 column dodges shared-memory bank conflicts
    int x = blockIdx.x*32 + threadIdx.x, y = blockIdx.y*32 + threadIdx.y;
    if ( x < inW && y < inH ) tile[threadIdx.y][threadIdx.x] = in[(size_t)y*inW + x];
    __syncthreads();
    int ox = blockIdx.y*32 + threadIdx.x, oy = blockIdx.x*32 + threadIdx.y;
    if ( ox < inH && oy < inW ) out[(size_t)oy*inH + ox] = tile[threadIdx.x][threadIdx.y];
}

// CUB iterators: load uint8 pixels widened to int64 (so the scan accumulates in
// int64, never overflowing), and key each element by its row index.
struct LoadU8 { const uint8_t *p; __host__ __device__ int64_t operator()( int64_t i ) const { return (int64_t) p[i]; } };
struct RowKey { int  w;           __host__ __device__ int64_t operator()( int64_t i ) const { return i / w; } };

// One WxW size: build the SAT both ways, time each, confirm both are exact.
static int
run( int W, const uint8_t *hImg, uint8_t *dImg,
     int64_t *dS, int64_t *dA, int64_t *dAt, int64_t *dBt, void *dTmp, size_t tmpBytes )
{
    cudaError_t status_cudart;
    int ret = 1;
    const int H = W;
    const size_t N = (size_t)W*H;
    int64_t *hS = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL;
    float msNaive = 0.f, msCub = 0.f;
    long naiveBad = 0, cubBad = 0;

    // declare all initialized locals before the first cuda() macro, whose
    // goto-on-error must not bypass an initialization.
    dim3 tblk( 32, 32 ), tgrid( (W+31)/32, (H+31)/32 ), tgridT( (H+31)/32, (W+31)/32 );
    thrust::counting_iterator<int64_t> cnt( 0 );
    auto rowKeysW = thrust::make_transform_iterator( cnt, RowKey{ W } );
    auto rowKeysH = thrust::make_transform_iterator( cnt, RowKey{ H } );
    auto pixels   = thrust::make_transform_iterator( cnt, LoadU8{ dImg } );

    cuda( EventCreate( &e0 ) );  cuda( EventCreate( &e1 ) );
    hS = (int64_t *) malloc( N*sizeof(int64_t) );

    // --- naive: serial row scan, then serial column scan ---
    #define NAIVE() do { \
        serialRow<<<(H+255)/256,256>>>( dImg, W, H, dS ); \
        serialCol<<<(W+255)/256,256>>>( W, H, dS ); } while (0)
    NAIVE();  cuda( GetLastError() );  cuda( DeviceSynchronize() );
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < 50; i++ ) NAIVE();
    cuda( EventRecord( e1, 0 ) );  cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msNaive, e0, e1 ) );  msNaive /= 50;
    #undef NAIVE

    // validate the naive table (host four-corner query vs brute-force box sums)
    cuda( Memcpy( hS, dS, N*sizeof(int64_t), cudaMemcpyDeviceToHost ) );
    #define SAT(x,y) ( ((x)<0 || (y)<0) ? (int64_t)0 : hS[(size_t)(y)*W + (x)] )
    for ( int s = 0; s < 4000; s++ ) {
        int x0 = rand()%W, y0 = rand()%H;
        int x1 = x0 + 1 + rand()%(W-x0), y1 = y0 + 1 + rand()%(H-y0);
        int64_t box = SAT(x1-1,y1-1) - SAT(x0-1,y1-1) - SAT(x1-1,y0-1) + SAT(x0-1,y0-1);
        int64_t ref = 0;
        for ( int y = y0; y < y1; y++ ) {
            for ( int x = x0; x < x1; x++ ) ref += hImg[(size_t)y*W + x];
        }
        if ( box != ref ) naiveBad++;
    }

    // --- CUB: segmented row scan, transpose, segmented row scan again, transpose ---
    #define CUBBUILD() do { \
        size_t tb = tmpBytes; \
        CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( dTmp, tb, rowKeysW, pixels, dA, (int)N ) ); \
        transpose<<<tgrid,tblk>>>( dA, dAt, W, H ); \
        tb = tmpBytes; \
        CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( dTmp, tb, rowKeysH, dAt, dBt, (int)N ) ); \
        transpose<<<tgridT,tblk>>>( dBt, dS, H, W ); } while (0)
    CUBBUILD();  cuda( GetLastError() );  cuda( DeviceSynchronize() );
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < 50; i++ ) CUBBUILD();
    cuda( EventRecord( e1, 0 ) );  cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msCub, e0, e1 ) );  msCub /= 50;
    #undef CUBBUILD

    cuda( Memcpy( hS, dS, N*sizeof(int64_t), cudaMemcpyDeviceToHost ) );
    for ( int s = 0; s < 4000; s++ ) {
        int x0 = rand()%W, y0 = rand()%H;
        int x1 = x0 + 1 + rand()%(W-x0), y1 = y0 + 1 + rand()%(H-y0);
        int64_t box = SAT(x1-1,y1-1) - SAT(x0-1,y1-1) - SAT(x1-1,y0-1) + SAT(x0-1,y0-1);
        int64_t ref = 0;
        for ( int y = y0; y < y1; y++ ) {
            for ( int x = x0; x < x1; x++ ) ref += hImg[(size_t)y*W + x];
        }
        if ( box != ref ) cubBad++;
    }
    #undef SAT

    printf( "  %5d  %9.4f  %9.4f   %5.1fx   %s\n", W, msNaive, msCub, msNaive/msCub,
            (naiveBad || cubBad) ? "MISMATCH" : "exact" );
    ret = 0;

Error_cudart:
    if ( e0 ) cudaEventDestroy( e0 );
    if ( e1 ) cudaEventDestroy( e1 );
    free( hS );
    return ret;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status_cudart;
    int ret = 1;
    const int sizes[] = { 512, 1024, 2048, 4096 };
    const int nSizes = (int) (sizeof(sizes)/sizeof(sizes[0]));
    const int Wmax = sizes[nSizes-1];
    const size_t Nmax = (size_t)Wmax*Wmax;

    uint8_t *hImg = NULL, *dImg = NULL;
    int64_t *dS = NULL, *dA = NULL, *dAt = NULL, *dBt = NULL;
    void *dTmp = NULL;

    srand( 5 );
    hImg = (uint8_t *) malloc( Nmax );
    for ( size_t i = 0; i < Nmax; i++ ) hImg[i] = (uint8_t)(rand() & 0xff);

    cuda( Malloc( &dImg, Nmax ) );
    cuda( Malloc( &dS,  Nmax*sizeof(int64_t) ) );
    cuda( Malloc( &dA,  Nmax*sizeof(int64_t) ) );
    cuda( Malloc( &dAt, Nmax*sizeof(int64_t) ) );
    cuda( Malloc( &dBt, Nmax*sizeof(int64_t) ) );
    cuda( Memcpy( dImg, hImg, Nmax, cudaMemcpyHostToDevice ) );

    // Size the CUB temp storage once, for the largest scan we will run.
    {
        size_t tb = 0;
        thrust::counting_iterator<int64_t> cnt( 0 );
        auto keys = thrust::make_transform_iterator( cnt, RowKey{ Wmax } );
        auto vals = thrust::make_transform_iterator( cnt, LoadU8{ dImg } );
        CUDART_CHECK( cub::DeviceScan::InclusiveSumByKey( NULL, tb, keys, vals, dA, (int)Nmax ) );
        cuda( Malloc( &dTmp, tb ) );

        cudaDeviceProp prop;  cuda( GetDeviceProperties( &prop, 0 ) );
        printf( "%s  int64 summed-area table build (naive serial vs cub::DeviceScan)\n\n", prop.name );
        printf( "  %5s  %9s  %9s   %6s\n", "size", "naive ms", "CUB ms", "speedup" );

        for ( int i = 0; i < nSizes; i++ ) {
            if ( run( sizes[i], hImg, dImg, dS, dA, dAt, dBt, dTmp, tb ) ) goto Error_cudart;
        }
    }
    ret = 0;

Error_cudart:
    cudaFree( dImg );  cudaFree( dS );  cudaFree( dA );  cudaFree( dAt );  cudaFree( dBt );
    cudaFree( dTmp );
    free( hImg );
    return ret;
}
