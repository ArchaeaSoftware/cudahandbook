/*
 *
 * Copyright (C) 2011-2026 by Archaea Software, LLC.
 *      All rights reserved.
 *
 */

#include <cstdint>

#include "scanDecoupledLookback.cuh"

template<class T>
__host__ __device__ bool
isOdd( T x )
{
    return x & 1;
}

//
// Single-pass stream compaction, built on decoupled look-back: emit the input
// elements that satisfy a predicate (here, the odd ones), compactly and in
// order. One thread block processes one tile in a single pass -- it scans the
// tile's 0/1 predicate flags, looks back to learn how many elements earlier
// tiles kept, and scatters its kept elements to out[]. The last tile writes the
// total to *outCount. This is the shape CUB's DeviceSelect uses.
//
template<class T>
__global__ void
streamCompact_odd_kernel(
    T *out,
    int *outCount,                    // total kept (written by the last tile)
    const T *in,
    volatile scanStatus *status,      // one descriptor per tile (SCAN_X-initialized)
    uint32_t *tileCounter,            // one global counter, 0-initialized
    uint32_t numTiles,
    size_t N )
{
    extern __shared__ int sPartials[];   // blockDim.x predicate flags
    __shared__ uint32_t s_tile;
    __shared__ int s_base;               // number of elements kept by earlier tiles

    if ( threadIdx.x == 0 )
        s_tile = atomicAdd( tileCounter, 1 );
    __syncthreads();
    const uint32_t tile = s_tile;
    const size_t gidx = (size_t) tile * blockDim.x + threadIdx.x;

    //
    // Evaluate the predicate, then inclusive-scan the 0/1 flags in shared
    // memory (Kogge-Stone). sPartials[blockDim.x-1] is the tile's keep count.
    //
    T value = (T) 0;
    int pred = 0;
    if ( gidx < N ) {
        value = in[gidx];
        pred = isOdd( value ) ? 1 : 0;
    }
    sPartials[threadIdx.x] = pred;
    __syncthreads();
    for ( int off = 1; off < blockDim.x; off <<= 1 ) {
        int add = ( threadIdx.x >= off ) ? sPartials[threadIdx.x - off] : 0;
        __syncthreads();
        sPartials[threadIdx.x] += add;
        __syncthreads();
    }
    const int aggregate = sPartials[blockDim.x - 1];

    //
    // Cooperative look-back over the per-tile keep counts: s_base is the number
    // of elements kept by every earlier tile -- this tile's base output index.
    //
    scanCoopLookback<int>( status, tile, aggregate, s_base );
    __syncthreads();

    //
    // Scatter. sPartials[threadIdx.x] is the inclusive keep count, so
    // sPartials[threadIdx.x]-1 is this element's index among the tile's kept
    // elements; s_base offsets it into the global output.
    //
    if ( gidx < N && pred )
        out[s_base + sPartials[threadIdx.x] - 1] = value;

    if ( tile == numTiles - 1 && threadIdx.x == 0 )
        *outCount = s_base + aggregate;
}

template<class T>
void
streamCompact_odd( T *out, int *outCount, const T *in, size_t N, int b )
{
    cudaError_t status;
    scanStatus *gStatus = 0;
    uint32_t *tileCounter = 0;

    if ( N == 0 )
        return;

    uint32_t numTiles = (uint32_t) ( ( N + b - 1 ) / b );

    cuda(Malloc( &gStatus, numTiles * sizeof(scanStatus) ) );
    cuda(Memset( gStatus, 0, numTiles * sizeof(scanStatus) ) );   // SCAN_X == 0
    cuda(Malloc( &tileCounter, sizeof(uint32_t) ) );
    cuda(Memset( tileCounter, 0, sizeof(uint32_t) ) );

    streamCompact_odd_kernel<T><<<numTiles, b, b * sizeof(int)>>>(
        out, outCount, in, gStatus, tileCounter, numTiles, N );

Error:
    cudaFree( gStatus );
    cudaFree( tileCounter );
}
