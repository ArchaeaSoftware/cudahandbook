/*
 *
 * scanWarpShuffle.cuh
 *
 * Header file for shuffle-based scan (warp size).
 * Requires gpu-architecture sm_30 or higher.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
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

#ifndef __SCAN_WARP_SHUFFLE_CUH__
#define __SCAN_WARP_SHUFFLE_CUH__

__device__ __forceinline__
int
scanWarpShuffle_step(int partial, int offset)
{
    int result;
    asm(
        "{.reg .u32 r0;"
         ".reg .pred p;"
         "shfl.up.b32 r0|p, %1, %2, 0;"
         "@p add.u32 r0, r0, %3;"
         "mov.u32 %0, r0;}"
        : "=r"(result) : "r"(partial), "r"(offset), "r"(partial));
    return result;
}

template <int levels>
__device__ __forceinline__
int
scanWarpShuffle(int mysum)
{
    for(int i = 0; i < levels; ++i)
        mysum = scanWarpShuffle_step(mysum, 1 << i);
    return mysum;
}

template <int levels>
__device__ __forceinline__
int 
exclusive_scan_warp_shfl(int mysum)
{
    const unsigned int lane   = threadIdx.x & 31;
    for(int i = 0; i < levels; ++i)
        mysum = scanWarpShuffle_step( mysum, 1 << i);
    mysum = __shfl_up(mysum, 1);
    return (lane) ? mysum : 0;
}

template <int levels>
__device__ __forceinline__
int
inclusive_scan_warp_shfl(int mysum)
{
    return mysum + exclusive_scan_warp_shfl<levels>(mysum);
}

template <int logBlockSize>
__device__
int
scanBlockShuffle(int val, const unsigned int idx)
{
    const unsigned int lane   = idx & 31;
    const unsigned int warpid = idx >> 5;
    __shared__ int sPartials[32];

    // Intra-warp scan in each warp
    val = scanWarpShuffle<5>(val);

    // Collect per-warp results
    if (lane == 31) sPartials[warpid] = val;
    __syncthreads();

    // Use first warp to scan per-warp results
    if (warpid == 0) {
        int t = sPartials[lane];
        t = scanWarpShuffle<logBlockSize-5>( t );
        sPartials[lane] = t;
    }

    __syncthreads();

    // Add scanned base sum for final result
    if (warpid > 0) {
        val += sPartials[warpid - 1];
    }
    return val;
}

template <int logBlockSize>
__device__
int
exclusive_scan_block(int val, const unsigned int idx)
{
    const unsigned int lane   = idx & 31;
    const unsigned int warpid = idx >> 5;
    __shared__ int sPartials[32];

    // Intra-warp scan in each warp
    val = scanWarpShuffle<5>(val);

    // Collect per-warp results
    if (lane == 31) sPartials[warpid] = val;
    __syncthreads();

    // Use first warp to scan per-warp results
    if (warpid == 0) {
        int t = sPartials[lane];
        t = scanWarpShuffle<logBlockSize-5>( t );
        sPartials[lane] = t;
    }
    __syncthreads();

    // Accumulate results
    if (warpid > 0) 
        val += sPartials[warpid - 1];
    __syncthreads();

    if ( lane==31 ) 
        sPartials[warpid] = val;

    __syncthreads();
    val = __shfl_up(val, 1);
    if ( lane ) {
        return val;
    }
    return warpid ? sPartials[warpid-1] : 0;
}

template <int logBlockSize>
__device__
int
inclusive_scan_block(int val, const unsigned int idx)
{
    return val + exclusive_scan_block<logBlockSize>(val,idx);
}

#endif
