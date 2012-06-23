/*
 *
 * reduction3WarpSynchronousTemplated.cuh
 *
 * Header for templated formulation of reduction in shared memory.
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
// templated implementation of reduction can be used to generate
// optimized specializations for different block sizes.
//

template<unsigned int numThreads>
__global__ void
Reduction3_kernel( int *out, const int *in, size_t N )
{
    extern __shared__ int shared_sum[];
    const unsigned int tid = threadIdx.x;
    int sum = 0;
    for ( size_t i = blockIdx.x*numThreads + tid;
          i < N;
          i += numThreads*gridDim.x )
    {
        sum += in[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

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
        if ( tid == 0 ) {
            out[blockIdx.x] = wsSum[0];
        }
    }
}

template<unsigned int numThreads>
void
Reduction3_template( int *answer, int *partial, 
                     const int *in, size_t N, 
                     int numBlocks )
{
    Reduction3_kernel<numThreads><<< 
        numBlocks, numThreads, numThreads*sizeof(int)>>>( 
            partial, in, N );
    Reduction3_kernel<numThreads><<< 
        1, numThreads, numThreads*sizeof(int)>>>( 
            answer, partial, numBlocks );
}

void
Reduction3( int *out, int *partial, 
            const int *in, size_t N, 
            int numBlocks, int numThreads )
{
    switch ( numThreads ) {
        case    1: return Reduction3_template<   1>( out, partial, in, N, numBlocks );
        case    2: return Reduction3_template<   2>( out, partial, in, N, numBlocks );
        case    4: return Reduction3_template<   4>( out, partial, in, N, numBlocks );
        case    8: return Reduction3_template<   8>( out, partial, in, N, numBlocks );
        case   16: return Reduction3_template<  16>( out, partial, in, N, numBlocks );
        case   32: return Reduction3_template<  32>( out, partial, in, N, numBlocks );
        case   64: return Reduction3_template<  64>( out, partial, in, N, numBlocks );
        case  128: return Reduction3_template< 128>( out, partial, in, N, numBlocks );
        case  256: return Reduction3_template< 256>( out, partial, in, N, numBlocks );
        case  512: return Reduction3_template< 512>( out, partial, in, N, numBlocks );
        case 1024: return Reduction3_template<1024>( out, partial, in, N, numBlocks );
    }
}
