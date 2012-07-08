/*
 *
 * reduction4Templated.cuh
 *
 * Header for templated, single-pass reduction implementation.
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

template<typename ReductionType, typename T, unsigned int numThreads>
__device__ void
Reduction4_LogStepShared( volatile ReductionType *out, volatile ReductionType *sPartials )
{
    const int tid = threadIdx.x;
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
    }
}

// Global variable used by reduceSinglePass to count blocks
__device__ unsigned int retirementCount = 0;

template<typename ReductionType, typename T, unsigned int numThreads>
__global__ void reduceSinglePass(ReductionType *out, ReductionType *partial, const T *in, unsigned int N)
{
    SharedMemory<ReductionType> sPartials;
    const unsigned int tid = threadIdx.x;
    ReductionType sum;
    for ( size_t i = blockIdx.x*numThreads + tid;
          i < N;
          i += numThreads*gridDim.x )
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    if (gridDim.x == 1) {
        Reduction4_LogStepShared<ReductionType, T, numThreads>( &out[blockIdx.x], sPartials );
        if ( tid==0 ) out[blockIdx.x] = sPartials[0];
        return;
    }
    Reduction4_LogStepShared<ReductionType, T, numThreads>( &partial[blockIdx.x], sPartials );
    if ( tid==0 ) partial[blockIdx.x] = sPartials[0];

    __shared__ bool lastBlock;

    // wait until all outstanding memory instructions in this thread are finished
    __threadfence();

    // Thread 0 takes a ticket
    if( tid==0 ) {
        unsigned int ticket = atomicAdd(&retirementCount, 1);
        // If the ticket ID is equal to the number of blocks, we are the last block!
        lastBlock = (ticket == gridDim.x-1);
    }
    __syncthreads();

    // One block performs the final log-step reduction
    if( lastBlock ) {
        ReductionType sum;
        for ( size_t i = threadIdx.x; i < gridDim.x; i += numThreads ) {
            sum += partial[i];
        }
        sPartials[threadIdx.x] = sum;
        __syncthreads();
        Reduction4_LogStepShared<ReductionType, T, numThreads>( out, sPartials );
        if ( tid==0 ) out[0] = sPartials[0];
        retirementCount = 0;
    }
}

template<typename ReductionType, typename T, unsigned int numThreads>
void
Reduction4_template( ReductionType *out, ReductionType *partial, const T *in, size_t N, int numBlocks )
{
    reduceSinglePass<ReductionType, T, numThreads><<< numBlocks, numThreads, numThreads*sizeof(ReductionType)>>>( out, partial, in, N );
}

template<typename ReductionType, typename T>
void 
Reduction4( ReductionType *out, ReductionType *partial, const T *in, size_t N, int numBlocks, int numThreads )
{
    if ( N < numBlocks*numThreads ) {
        numBlocks = (N+numThreads-1)/numThreads;
    }
    switch (numThreads) {
        case 512: Reduction4_template<ReductionType, T, 512>(out, partial, in, N, numBlocks); break;
        case 256: Reduction4_template<ReductionType, T, 256>(out, partial, in, N, numBlocks); break;
        case 128: Reduction4_template<ReductionType, T, 128>(out, partial, in, N, numBlocks); break;
        case  64: Reduction4_template<ReductionType, T,  64>(out, partial, in, N, numBlocks); break;
        case  32: Reduction4_template<ReductionType, T,  32>(out, partial, in, N, numBlocks); break;
        case  16: Reduction4_template<ReductionType, T,  16>(out, partial, in, N, numBlocks); break;
        case   8: Reduction4_template<ReductionType, T,   8>(out, partial, in, N, numBlocks); break;
        case   4: Reduction4_template<ReductionType, T,   4>(out, partial, in, N, numBlocks); break;
        case   2: Reduction4_template<ReductionType, T,   2>(out, partial, in, N, numBlocks); break;
        case   1: Reduction4_template<ReductionType, T,   1>(out, partial, in, N, numBlocks); break;
    }
}
