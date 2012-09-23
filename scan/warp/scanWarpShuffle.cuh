/*
 *
 * scanWarpShuffle.cuh
 *
 * Header file for shuffle-based scan (warp size).
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

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

__device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
{
    uint result;
    asm(
        "{.reg .u32 r0;"
         ".reg .pred p;"
         "shfl.up.b32 r0|p, %1, %2, 0;"
         "@p add.u32 r0, r0, %3;"
         "mov.u32 %0, r0;}"
        : "=r"(result) : "r"(partial), "r"(up_offset), "r"(partial));
    return result;
}

template <int levels>
__device__ __forceinline__ uint inclusive_scan_warp_shfl(int mysum)
{
    // this pragma may be unnecessary with the template parameter!
    #pragma unroll
    for(int i = 0; i < levels; ++i)
        mysum = shfl_scan_add_step(mysum, 1 << i);
    return mysum;
}

template <int levels>
__device__ __forceinline__ uint exclusive_scan_warp_shfl(int mysum)
{
    // this pragma may be unnecessary with the template parameter!
    #pragma unroll
    for(int i = 0; i < levels; ++i)
        mysum = shfl_scan_add_step(mysum, 1 << i);
    mysum = __shfl_up(mysum, 1);
    return (threadIdx.x&31) ? mysum : 0;
}

template <int logBlockSize>
__device__ uint inclusive_scan_block(uint val, const unsigned int idx)
{
    const unsigned int lane   = idx & 31;
    const unsigned int warpid = idx >> 5;
    __shared__ uint ptr[WARP_SIZE];

    // step 1: Intra-warp scan in each warp

    val = inclusive_scan_warp_shfl<LOG_WARP_SIZE>(val);

    // step 2: Collect per-warp particle results
    if (lane == 31) ptr[warpid] = val;
    __syncthreads();
    // step 3: Use 1st warp to scan per-warp results
    if (warpid == 0) ptr[lane] = inclusive_scan_warp_shfl<logBlockSize-LOG_WARP_SIZE>(ptr[lane]);
    __syncthreads();
    // step 4: Accumulate results from Steps 1 and 3;
    if (warpid > 0) val += ptr[warpid - 1];
    // __syncthreads(); // MJH don't think this sync is needed since we have a function-scope
    // shared memory array. But you might want to use a shared allocation that gets reused later
    // to cut down shared  usage even further.  In that case either sync there or before you
    // write to the aliased shared memory addresses upon returning from this call.
    return val;
}

#endif
