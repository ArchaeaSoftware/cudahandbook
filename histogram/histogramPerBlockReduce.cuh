/*
 *
 * histogramPerBlockReduce.cuh
 *
 * Implementation of histogram that uses one shared atomic per pixel,
 * then writes the shared histogram to a temporary buffer that is
 * then reduced to the output histogram by a subsequent kernel launch.
 * This method is reminiscent of early methods of doing Scan in CUDA.
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

__device__ unsigned int tempHist[400][256];

template<bool bOffset>
__global__ void
histogram1DPerBlockReduce(
    unsigned int *pHist,
    const unsigned char *base, size_t N )
{
    __shared__ unsigned int sHist[256];
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        sHist[i] = 0;
    }
    __syncthreads();
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N/4;
              i += blockDim.x*gridDim.x ) {
        unsigned int value = ((unsigned int *) base)[i];
        incPerBlockHistogram<bOffset>( sHist, value & 0xff ); value >>= 8;
        incPerBlockHistogram<bOffset>( sHist, value & 0xff ); value >>= 8;
        incPerBlockHistogram<bOffset>( sHist, value & 0xff ); value >>= 8;
        incPerBlockHistogram<bOffset>( sHist, value & 0xff ); 
    }
    __syncthreads();
    unsigned int *outputHist = &tempHist[blockIdx.x][0];
    for ( int i = threadIdx.x; i < 256; i += blockDim.x*blockDim.y ) {
        int srcIndex = (bOffset) ? 0xff&(i+blockIdx.x) : i;
        outputHist[i] = sHist[srcIndex];
    }
}

// Must be invoked with 256 threads, one for each output histogram element
__global__ void
histogramPerBlockFinalReduction( 
    unsigned int *pHist, 
    unsigned int cHist )
{
    unsigned int sum = 0;
    for ( int i = 0; i < cHist; i++ ) {
        sum += tempHist[i][threadIdx.x];
    }
    pHist[threadIdx.x] = sum;
}

template<bool bOffset>
void
GPUhistogramPerBlockReduce(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    cudaError_t status;
    cudaEvent_t start = 0, stop = 0;
    

    cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );

    cuda(EventRecord( start, 0 ) );
    {
        void *ptempHist;
        cuda(GetSymbolAddress( &ptempHist, tempHist ) );
        cuda(Memset( ptempHist, 0, sizeof(tempHist ) ) );
    }
    histogram1DPerBlockReduce<bOffset><<<240,threads.x*threads.y>>>( pHist, dptrBase, w*h );
    histogramPerBlockFinalReduction<<<1,256>>>( pHist, 240 );
    
    cuda(EventRecord( stop, 0 ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}

void
GPUhistogramPerBlockReduce(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerBlockReduce<false>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}

void
GPUhistogramPerBlockReduceOffset(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerBlockReduce<true>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}
