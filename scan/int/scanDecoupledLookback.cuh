/*
 *
 * scanDecoupledLookback.cuh
 *
 * Single-pass inclusive scan with decoupled look-back, after
 * Merrill & Garland, "Single-pass Parallel Prefix Scan with Decoupled
 * Look-back," NVIDIA Technical Report NVR-2016-002 (2016) -- the method
 * used by CUB/Thrust's DeviceScan.
 *
 * One thread block processes one tile of blockDim.x elements in a single
 * pass over the input. Each tile publishes a status descriptor; a tile
 * computes its exclusive prefix by looking back over predecessors, summing
 * their aggregates until it reaches one whose inclusive prefix is already
 * available -- reusing that prefix instead of recomputing it. The look-back
 * is cooperative: one warp inspects 32 predecessors at a time, using __ballot
 * to find the nearest ready prefix and a warp reduction to sum the aggregates
 * back to it (after CUB).
 *
 * The descriptor packs the status flag and its value into one 64-bit word
 * so the two are written and read atomically together; no separate memory
 * fence is needed. This costs one bit of generality: it assumes a 32-bit
 * element type and a commutative, associative operator (here, integer
 * addition). A 64-bit element type needs a two-word descriptor and a fence,
 * as CUB implements.
 *
 * Copyright (c) 2016-2026, Archaea Software, LLC.
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
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
 *
 */

#ifndef __SCAN_DECOUPLED_LOOKBACK_CUH__
#define __SCAN_DECOUPLED_LOOKBACK_CUH__

#include <cstdint>

//
// A tile's status descriptor: the flag in the high 32 bits, its value in the
// low 32 bits, so a reader that observes the flag also observes the matching
// value. Descriptors are initialized to SCAN_X (all zero) by cudaMemset.
//
enum { SCAN_X = 0, SCAN_A = 1, SCAN_P = 2 };  // invalid / aggregate / prefix

// 64-bit descriptor word. Deliberately unsigned long long, not uint64_t: CUDA's
// 64-bit atomics (atomicOr/atomicExch) are overloaded only for unsigned long
// long, and uint64_t is a distinct type (unsigned long on LP64) that won't bind.
typedef unsigned long long scanStatus;

__device__ __forceinline__ scanStatus
scanPackStatus( uint32_t flag, int value )
{
    return ( (scanStatus) flag << 32 ) | (uint32_t) value;
}

//
// Warp-cooperative look-back: warp 0 computes this tile's exclusive prefix into
// s_base, and publishes this tile's inclusive prefix. Shared by the decoupled
// look-back scan (both variants) and stream compaction. Reads 32 predecessors
// at a time; __ballot finds the nearest ready prefix, a warp reduction sums the
// aggregates back to it. The kernel below inlines the same logic to show it in
// full; every other caller uses this function.
//
template<class T>
__device__ __forceinline__ void
scanCoopLookback( volatile scanStatus *status_cudart, uint32_t tile, T aggregate, T &s_base )
{
    if ( tile == 0 ) {
        if ( threadIdx.x == 0 ) {
            atomicExch( (scanStatus *) &status_cudart[tile], scanPackStatus( SCAN_P, aggregate ) );
            s_base = (T) 0;
        }
        return;
    }
    if ( threadIdx.x == 0 )
        atomicExch( (scanStatus *) &status_cudart[tile], scanPackStatus( SCAN_A, aggregate ) );
    if ( threadIdx.x < 32 ) {
        const int lane = threadIdx.x;
        T exclusive = (T) 0;
        int frontier = (int) tile - 1;
        for ( ;; ) {
            const int pidx = frontier - lane;
            uint32_t flag; T val;
            do {
                if ( pidx >= 0 ) {
                    scanStatus d = atomicOr( (scanStatus *) &status_cudart[pidx], 0ull );
                    flag = (uint32_t) ( d >> 32 );
                    val  = (T) (int) ( d & 0xffffffffu );
                } else { flag = SCAN_P; val = (T) 0; }
            } while ( __any_sync( 0xffffffffu, flag == SCAN_X ) );
            const uint32_t pmask = __ballot_sync( 0xffffffffu, flag == SCAN_P );
            const int firstP = pmask ? ( __ffs( pmask ) - 1 ) : 32;
            T contrib = ( lane <= firstP ) ? val : (T) 0;
            #pragma unroll
            for ( int off = 16; off > 0; off >>= 1 )
                contrib += __shfl_xor_sync( 0xffffffffu, contrib, off );
            exclusive += contrib;
            if ( firstP < 32 ) break;
            frontier -= 32;
        }
        if ( lane == 0 ) {
            s_base = exclusive;
            atomicExch( (scanStatus *) &status_cudart[tile],
                        scanPackStatus( SCAN_P, exclusive + aggregate ) );
        }
    }
}

template<class T>
__global__ void
scanDecoupledLookback_kernel(
    T *out,
    const T *in,
    volatile scanStatus *status_cudart,  // one descriptor per tile (SCAN_X-initialized)
    uint32_t *tileCounter,          // one global counter, 0-initialized
    size_t N )
{
    extern __shared__ T s[];        // blockDim.x elements
    __shared__ uint32_t s_tile;
    __shared__ T s_exclusive;

    //
    // Claim a tile index from a global counter. A block that later waits on
    // predecessor p is then guaranteed p is already resident and running (p
    // claimed a smaller index earlier), so the look-back cannot deadlock.
    //
    if ( threadIdx.x == 0 ) {
        s_tile = atomicAdd( tileCounter, 1 );
    }
    __syncthreads();
    const uint32_t tile = s_tile;
    const size_t base = (size_t) tile * blockDim.x;
    const size_t gidx = base + threadIdx.x;

    //
    // Load the tile (zero-filling past the end) and inclusive-scan it in shared
    // memory (Kogge-Stone). s[blockDim.x-1] then holds the tile's aggregate.
    //
    s[threadIdx.x] = ( gidx < N ) ? in[gidx] : (T) 0;
    __syncthreads();
    for ( int off = 1; off < blockDim.x; off <<= 1 ) {
        T add = ( threadIdx.x >= off ) ? s[threadIdx.x - off] : (T) 0;
        __syncthreads();
        s[threadIdx.x] += add;
        __syncthreads();
    }
    const T aggregate = s[blockDim.x - 1];

    //
    // Compute this tile's exclusive prefix. Tile 0 has no predecessors; every
    // other tile advertises its aggregate, then warp 0 looks back cooperatively
    // over 32 predecessors at a time.
    //
    if ( tile == 0 ) {
        if ( threadIdx.x == 0 ) {
            atomicExch( (scanStatus *) &status_cudart[tile],
                        scanPackStatus( SCAN_P, aggregate ) );
            s_exclusive = (T) 0;
        }
    }
    else {
        // Advertise our aggregate so successors can make progress without us.
        if ( threadIdx.x == 0 )
            atomicExch( (scanStatus *) &status_cudart[tile],
                        scanPackStatus( SCAN_A, aggregate ) );

        if ( threadIdx.x < 32 ) {
            const int lane = threadIdx.x;
            T exclusive = (T) 0;
            int frontier = (int) tile - 1;      // nearest predecessor not yet consumed
            for ( ;; ) {
                // Lane L inspects predecessor (frontier - L): the 32 nearest
                // predecessors at once, lane 0 == nearest.
                const int pidx = frontier - lane;
                uint32_t flag;
                T val;
                do {                            // re-read the window until no lane sees X
                    if ( pidx >= 0 ) {
                        scanStatus d = atomicOr( (scanStatus *) &status_cudart[pidx], 0ull );
                        flag = (uint32_t) ( d >> 32 );
                        val  = (T) (int) ( d & 0xffffffffu );
                    } else {
                        flag = SCAN_P;          // out of range: prefix 0, stops the walk
                        val  = (T) 0;
                    }
                } while ( __any_sync( 0xffffffffu, flag == SCAN_X ) );

                // Nearest inclusive prefix in the window (lowest lane with P).
                const uint32_t pmask = __ballot_sync( 0xffffffffu, flag == SCAN_P );
                const int firstP = pmask ? ( __ffs( pmask ) - 1 ) : 32;

                // Sum lanes 0..firstP: aggregates up to it, plus its prefix.
                T contrib = ( lane <= firstP ) ? val : (T) 0;
                #pragma unroll
                for ( int off = 16; off > 0; off >>= 1 )
                    contrib += __shfl_xor_sync( 0xffffffffu, contrib, off );
                exclusive += contrib;

                if ( firstP < 32 )              // reached a prefix -> done
                    break;
                frontier -= 32;                 // whole window was aggregates -> keep walking
            }
            if ( lane == 0 ) {
                s_exclusive = exclusive;
                atomicExch( (scanStatus *) &status_cudart[tile],
                            scanPackStatus( SCAN_P, exclusive + aggregate ) );
            }
        }
    }
    __syncthreads();

    //
    // Add the exclusive prefix to the local inclusive scan and write the output.
    //
    if ( gidx < N ) {
        out[gidx] = s_exclusive + s[threadIdx.x];
    }
}

template<class T>
void
scanDecoupledLookback( T *out, const T *in, size_t N, int b )
{
    cudaError_t status_cudart;
    scanStatus *gStatus = 0;
    uint32_t *tileCounter = 0;

    if ( N == 0 )
        return;

    uint32_t numTiles = (uint32_t) ( ( N + b - 1 ) / b );

    cuda(Malloc( &gStatus, numTiles * sizeof(scanStatus) ) );
    cuda(Memset( gStatus, 0, numTiles * sizeof(scanStatus) ) );   // SCAN_X == 0
    cuda(Malloc( &tileCounter, sizeof(uint32_t) ) );
    cuda(Memset( tileCounter, 0, sizeof(uint32_t) ) );

    scanDecoupledLookback_kernel<T><<<numTiles, b, b * sizeof(T)>>>(
        out, in, gStatus, tileCounter, N );

Error_cudart:
    cudaFree( gStatus );
    cudaFree( tileCounter );
}

#endif // __SCAN_DECOUPLED_LOOKBACK_CUH__
