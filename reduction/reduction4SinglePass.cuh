/*
 *
 * reduction4SinglePass.cuh
 *
 * Header for single-pass formulation of reduction in shared memory.
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
// single-pass implementation of reduction uses this subroutine
//
template<unsigned int numThreads>
__device__ void
Reduction4_LogStepShared( int *out, volatile int *partials )
{
    const int tid = threadIdx.x;
    if (numThreads >= 1024) {
        if (tid < 512) { 
            partials[tid] += partials[tid + 512];
        }
        __syncthreads();
    }
    if (numThreads >= 512) {
        if (tid < 256) { 
            partials[tid] += partials[tid + 256];
        }
        __syncthreads();
    }
    if (numThreads >= 256) {
        if (tid < 128) {
            partials[tid] += partials[tid + 128];
        }
        __syncthreads();
    }
    if (numThreads >= 128) {
        if (tid <  64) {
            partials[tid] += partials[tid +  64];
        }
        __syncthreads();
    }

    // warp synchronous at the end
    if ( tid < 32 ) {
        if (numThreads >=  64) { partials[tid] += partials[tid + 32]; }
        if (numThreads >=  32) { partials[tid] += partials[tid + 16]; }
        if (numThreads >=  16) { partials[tid] += partials[tid +  8]; }
        if (numThreads >=   8) { partials[tid] += partials[tid +  4]; }
        if (numThreads >=   4) { partials[tid] += partials[tid +  2]; }
        if (numThreads >=   2) { partials[tid] += partials[tid +  1]; }
        if ( tid == 0 ) {
            *out = partials[0];
        }
    }
}

// Global variable used by reduceSinglePass to count blocks
__device__ unsigned int retirementCount = 0;

template <unsigned int numThreads>
__global__ void 
reduceSinglePass( int *out, int *partial, 
                  const int *in, unsigned int N )
{
    extern __shared__ int sPartials[];
    unsigned int tid = threadIdx.x;
    int sum = 0;
    for ( size_t i = blockIdx.x*numThreads + tid;
                 i < N;
                 i += numThreads*gridDim.x ) {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    if (gridDim.x == 1) {
        Reduction4_LogStepShared<numThreads>( &out[blockIdx.x], 
                                              sPartials );
        return;
    }
    Reduction4_LogStepShared<numThreads>( &partial[blockIdx.x], 
                                          sPartials );

    __shared__ bool lastBlock;

    // wait for outstanding memory instructions in this thread
    __threadfence();

    // Thread 0 takes a ticket
    if( tid==0 ) {
        unsigned int ticket = atomicAdd(&retirementCount, 1);
        
        //
        // If the ticket ID is equal to the number of blocks, 
        // we are the last block!
        //
        lastBlock = (ticket == gridDim.x-1);
    }
    __syncthreads();

    // One block performs the final log-step reduction
    if( lastBlock ) {
        int sum = 0;
        for ( size_t i = tid; 
                     i < gridDim.x; 
                     i += numThreads ) {
            sum += partial[i];
        }
        sPartials[threadIdx.x] = sum;
        __syncthreads();
        Reduction4_LogStepShared<numThreads>( out, sPartials );
        retirementCount = 0;
    }
}

template<unsigned int numThreads>
void
Reduction4_template( int *out, int *partial, 
                     const int *in, size_t N, 
                     int numBlocks )
{
    reduceSinglePass<numThreads><<< 
        numBlocks, numThreads, numThreads*sizeof(int)>>>( 
            out, partial, in, N );
}

//
// generate the template specializations
//
void 
Reduction4( int *out, int *partial, 
            const int *in, size_t N, 
            int numBlocks, int numThreads )
{
    switch (numThreads) {
        case 1024: Reduction4_template<1024>(out, partial, in, N, numBlocks); break;
        case  512: Reduction4_template< 512>(out, partial, in, N, numBlocks); break;
        case  256: Reduction4_template< 256>(out, partial, in, N, numBlocks); break;
        case  128: Reduction4_template< 128>(out, partial, in, N, numBlocks); break;
        case   64: Reduction4_template<  64>(out, partial, in, N, numBlocks); break;
        case   32: Reduction4_template<  32>(out, partial, in, N, numBlocks); break;
        case   16: Reduction4_template<  16>(out, partial, in, N, numBlocks); break;
        case    8: Reduction4_template<   8>(out, partial, in, N, numBlocks); break;
        case    4: Reduction4_template<   4>(out, partial, in, N, numBlocks); break;
        case    2: Reduction4_template<   2>(out, partial, in, N, numBlocks); break;
        case    1: Reduction4_template<   1>(out, partial, in, N, numBlocks); break;
    }
}
