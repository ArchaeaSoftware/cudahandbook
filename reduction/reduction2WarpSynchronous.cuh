/*
 *
 * reduction2WarpSynchronous.cuh
 *
 * Header for warp synchronous formulation of reduction.
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

__global__ void
Reduction2_kernel( int *out, const int *in, size_t N )
{
    extern __shared__ int sPartials[];
    int sum = 0;
    const int tid = threadIdx.x;
    for ( size_t i = blockIdx.x*blockDim.x + tid;
          i < N;
          i += blockDim.x*gridDim.x ) {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    for ( int activeThreads = blockDim.x>>1; 
              activeThreads > 32; 
              activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
            sPartials[tid] += sPartials[tid+activeThreads];
        }
        __syncthreads();
    }
    if ( threadIdx.x < 32 ) {
        volatile int *wsSum = sPartials;
        if ( blockDim.x > 32 ) wsSum[tid] += wsSum[tid + 32];
        wsSum[tid] += wsSum[tid + 16];
        wsSum[tid] += wsSum[tid + 8];
        wsSum[tid] += wsSum[tid + 4];
        wsSum[tid] += wsSum[tid + 2];
        wsSum[tid] += wsSum[tid + 1];
        if ( tid == 0 ) {
            volatile int *wsSum = sPartials;
            out[blockIdx.x] = wsSum[0];
        }
    }
}

void
Reduction2( int *answer, int *partial, 
            const int *in, size_t N, 
            int numBlocks, int numThreads )
{
    Reduction2_kernel<<< 
        numBlocks, numThreads, numThreads*sizeof(int)>>>( 
            partial, in, N );
    Reduction2_kernel<<<
        1, numThreads, numThreads*sizeof(int)>>>( 
            answer, partial, numBlocks );
}
