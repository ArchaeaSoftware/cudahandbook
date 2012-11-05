/*
 *
 * reduction5Templated.cuh
 *
 * CUDA header for templated formulation of reduction.
 *
 * Build with: nvcc -I ../chLib <options> reduction5Templated.cuh
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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
// numThreads must be a power of 2!
//

template<typename ReductionType, typename T, unsigned int numThreads, unsigned int numBlocks>
__global__ void
Reduction5_kernel( ReductionType *out, const T *in, size_t N )
{
    SharedMemory<ReductionType> sPartials;
    const unsigned int tid = threadIdx.x;
    ReductionType sum;
    for ( size_t i = blockIdx.x*numThreads + tid;
          i < N;
          i += numThreads*numBlocks/*gridDim.x*/ )
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    if (numThreads >= 1024) { 
        if (tid < 512) {
            sPartials[tid] += sPartials[tid + 512];
        } 
        __syncthreads();
    }
    if (numThreads >= 512) { 
        if (tid < 256) {
            sPartials[tid] += sPartials[tid + 256];
        } 
        __syncthreads();
    }
    if (numThreads >= 256) {
        if (tid < 128) {
            sPartials[tid] += sPartials[tid + 128];
        } 
        __syncthreads();
    }
    if (numThreads >= 128) {
        if (tid <  64) { 
            sPartials[tid] += sPartials[tid +  64];
        } 
        __syncthreads();
    }
    // warp synchronous at the end
    if ( tid < 32 ) {
        volatile ReductionType *wsSum = sPartials;
        if (numThreads >=  64) { wsSum[tid] += wsSum[tid + 32]; }
        if (numThreads >=  32) { wsSum[tid] += wsSum[tid + 16]; }
        if (numThreads >=  16) { wsSum[tid] += wsSum[tid +  8]; }
        if (numThreads >=   8) { wsSum[tid] += wsSum[tid +  4]; }
        if (numThreads >=   4) { wsSum[tid] += wsSum[tid +  2]; }
        if (numThreads >=   2) { wsSum[tid] += wsSum[tid +  1]; }
        if ( tid == 0 ) {
            out[blockIdx.x] = sPartials[0];
        }
    }
}

template<typename ReductionType, typename T, unsigned int numThreads>
void
Reduction5_template( ReductionType *answer, ReductionType *partial, const T *in, size_t N, int numBlocks )
{
    Reduction5_kernel<ReductionType, T, numThreads, 120><<< 120/*numBlocks*/, numThreads, numThreads*sizeof(ReductionType)>>>( partial, in, N );
    Reduction5_kernel<ReductionType, ReductionType, numThreads, 1><<<         1, numThreads, numThreads*sizeof(ReductionType)>>>( answer, partial, numBlocks );
}

template<typename ReductionType, typename T>
void
Reduction5( ReductionType *out, ReductionType *partial, const T *in, size_t N, int numBlocks, int numThreads )
{
    if ( N < numBlocks*numThreads ) {
        numBlocks = (N+numThreads-1)/numThreads;
    }
    switch ( numThreads ) {
        case    1: return Reduction5_template<ReductionType, T,   1>( out, partial, in, N, numBlocks );
        case    2: return Reduction5_template<ReductionType, T,   2>( out, partial, in, N, numBlocks );
        case    4: return Reduction5_template<ReductionType, T,   4>( out, partial, in, N, numBlocks );
        case    8: return Reduction5_template<ReductionType, T,   8>( out, partial, in, N, numBlocks );
        case   16: return Reduction5_template<ReductionType, T,  16>( out, partial, in, N, numBlocks );
        case   32: return Reduction5_template<ReductionType, T,  32>( out, partial, in, N, numBlocks );
        case   64: return Reduction5_template<ReductionType, T,  64>( out, partial, in, N, numBlocks );
        case  128: return Reduction5_template<ReductionType, T, 128>( out, partial, in, N, numBlocks );
        case  256: return Reduction5_template<ReductionType, T, 256>( out, partial, in, N, numBlocks );
        case  512: return Reduction5_template<ReductionType, T, 512>( out, partial, in, N, numBlocks );
        case 1024: return Reduction5_template<ReductionType, T,1024>( out, partial, in, N, numBlocks );
    }
}
