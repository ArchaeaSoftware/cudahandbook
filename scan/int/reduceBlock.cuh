/*
 *
 * reduceBlock.cuh
 *
 * Utility device function to compute the reduction across a block.
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

#ifndef __CUDAHANDBOOK_REDUCE_BLOCK__
#define __CUDAHANDBOOK_REDUCE_BLOCK__

template<class T, int numThreads>
__device__ void
reduceBlock( T *globalSum, volatile T *shared_sum )
{
    const int tid = threadIdx.x;
    if (numThreads >= 1024) { 
        if (tid < 512) { 
            shared_sum[tid] += shared_sum[tid + 512]; 
        } 
        __syncthreads();
    }
    if (numThreads >= 512) { 
        if (tid < 256) { 
            shared_sum[tid] += shared_sum[tid + 256]; 
        } 
        __syncthreads();
    }
    if (numThreads >= 256) {
        if (tid < 128) {
            shared_sum[tid] += shared_sum[tid + 128];
        } 
        __syncthreads();
    }
    if (numThreads >= 128) {
        if (tid <  64) { 
            shared_sum[tid] += shared_sum[tid +  64];
        } 
        __syncthreads();
    }

    // warp synchronous at the end
    if ( tid < 32 ) {
        volatile int *wsSum = shared_sum;
        if (numThreads >=  64) { wsSum[tid] += wsSum[tid + 32]; }
        if (numThreads >=  32) { wsSum[tid] += wsSum[tid + 16]; }
        if (numThreads >=  16) { wsSum[tid] += wsSum[tid +  8]; }
        if (numThreads >=   8) { wsSum[tid] += wsSum[tid +  4]; }
        if (numThreads >=   4) { wsSum[tid] += wsSum[tid +  2]; }
        if (numThreads >=   2) { wsSum[tid] += wsSum[tid +  1]; }
        if ( tid == 0 ) *globalSum = wsSum[0];
    }
}

#endif //__CUDAHANDBOOK_REDUCE_BLOCK__
