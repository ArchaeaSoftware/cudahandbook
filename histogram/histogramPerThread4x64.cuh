/*
 *
 * histogramPerThread4x64.cuh
 *
 * Implementation of histogram that uses 8-bit privatized counters
 * and uses 32-bit increments to process them.
 * This version is the same as histogramPerThread64.cuh,
 * but unrolls the inner loop 4x.  That requires the number of blocks
 * to be adjusted upward, accordingly, to keep the counters from
 * rolling over.
 *
 * Requires: SM 1.2, for shared atomics.
 *
 * Copyright (c) 2013, Archaea Software, LLC.
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

template<bool bPeriodicMerge>
__global__ void
histogram1DPerThread4x64(
    unsigned int *pHist,
    const unsigned char *base, size_t N )
{
    extern __shared__ unsigned int privHist[];
    const int blockDimx = 64;

    if ( blockDim.x != blockDimx ) return;
    
    for ( int i = threadIdx.x;
              i < 64*blockDimx;
              i += blockDimx ) {
        privHist[i] = 0;
    }
    __syncthreads();
    int cIterations = 0;
    for ( int i = blockIdx.x*blockDimx+threadIdx.x;
              i < N/4;
              i += blockDimx*gridDim.x ) {
        unsigned int value = ((unsigned int *) base)[i];
        incPrivatized32Element( value & 0xff ); value >>= 8;
        incPrivatized32Element( value & 0xff ); value >>= 8;
        incPrivatized32Element( value & 0xff ); value >>= 8;
        incPrivatized32Element( value );
        cIterations += 1;
        if ( bPeriodicMerge && cIterations>=252/4 ) {
            cIterations = 0;
            __syncthreads();
            merge64HistogramsToOutput<true>( pHist );
        }
    }
    __syncthreads();

    merge64HistogramsToOutput<false>( pHist );
}

template<bool bPeriodicMerge>
void
GPUhistogramPerThread4x64(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    cudaError_t status;
    cudaEvent_t start = 0, stop = 0;
    int numthreads = threads.x*threads.y;
    int numblocks = bPeriodicMerge ? 256 : INTDIVIDE_CEILING( w*h, numthreads*(255/4) );

    cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );

    cuda(Memset( pHist, 0, 256*sizeof(unsigned int) ) );

    cuda(EventRecord( start, 0 ) );
    histogram1DPerThread4x64<bPeriodicMerge><<<numblocks,numthreads,numthreads*256>>>( pHist, dptrBase, w*h );
    cuda(EventRecord( stop, 0 ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}

void
GPUhistogramPerThread4x64(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerThread4x64<false>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}

void
GPUhistogramPerThread4x64_PeriodicMerge(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerThread4x64<true>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}
