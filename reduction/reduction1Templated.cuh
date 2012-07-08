/*
 * reduction1Templated.cuh
 *
 * Header for simplest formulation of templated reduction.
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

template<typename ReductionType, typename T>
__global__ void
Reduction1_kernel( ReductionType *out, const T *in, size_t N )
{
    SharedMemory<ReductionType> sPartials;
    ReductionType sum;
    const int tid = threadIdx.x;
    for ( size_t i = blockIdx.x*blockDim.x + tid;
          i < N;
          i += blockDim.x*gridDim.x ) {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    for ( int activeThreads = blockDim.x>>1; 
              activeThreads; 
              activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
            sPartials[tid] += sPartials[tid+activeThreads];
        }
        __syncthreads();
    }
    if ( tid == 0 ) {
        out[blockIdx.x] = sPartials[0];
    }
}

template<typename ReductionType, typename T>
void
Reduction1( ReductionType *answer, ReductionType *partial, const T *in, size_t N, int numBlocks, int numThreads )
{
    if ( N < numBlocks*numThreads ) {
        numBlocks = (N+numThreads-1)/numThreads;
    }
    Reduction1_kernel<ReductionType, T><<< 
        numBlocks, numThreads, numThreads*sizeof(ReductionType)>>>( 
            partial, in, N );
    Reduction1_kernel<ReductionType, ReductionType><<<         
        1, numThreads, numThreads*sizeof(ReductionType)>>>( 
            answer, partial, numBlocks );
}
