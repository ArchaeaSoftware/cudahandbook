/*
 *
 * reduction1ExplicitLoop.cuh
 *
 * Header for simplest formulation of reduction in shared memory.
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

//
// reads N ints and writes an intermediate sum per block
// blockDim.x must be a power of 2!
//
__global__ void
Reduction1_kernel( int *out, const int *in, size_t N )
{
    extern __shared__ int shared_sum[];
    int sum = 0;
    const int tid = threadIdx.x;
    for ( size_t i = blockIdx.x*blockDim.x + tid;
          i < N;
          i += blockDim.x*gridDim.x ) {
        sum += in[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

#pragma unroll 2
    for ( int activeThreads = blockDim.x>>1; 
              activeThreads; 
              activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
            shared_sum[tid] += shared_sum[tid+activeThreads];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {
        out[blockIdx.x] = shared_sum[0];
    }
}

void
Reduction1( int *answer, int *partial, 
            const int *in, size_t N, 
            int numBlocks, int numThreads )
{
    unsigned int sharedSize = numThreads*sizeof(int);
    Reduction1_kernel<<< 
        numBlocks, numThreads, sharedSize>>>( 
            partial, in, N );
    Reduction1_kernel<<< 
        1, numThreads, sharedSize>>>( 
            answer, partial, numBlocks );
}
