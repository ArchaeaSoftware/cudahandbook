/*
 *
 * histogramPrivatizedPerBlockReduce.cuh
 *
 * Implementation of histogram that uses one shared atomic per pixel,
 * then writes the shared histogram to a temporary buffer that is
 * then reduced to the output histogram by a subsequent kernel launch.
 * This method is reminiscent of early methods of doing Scan in CUDA.
 *
 * Requires: SM 1.2, for shared atomics.
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

__device__ unsigned int tempHist[400][256];

__global__ void
histogramPrivatizedPerBlockReduce( 
    unsigned int *pHist, 
    int x, int y, 
    int w, int h )
{
    __shared__ int sHist[256];
    const int tid = threadIdx.y*blockDim.x+threadIdx.x;
    for ( int i = tid; i < 256; i += blockDim.x*blockDim.y ) {
        sHist[i] = 0;
    }
    __syncthreads();
    for ( int row = blockIdx.y*blockDim.y+threadIdx.y; 
              row < h;
              row += blockDim.y*gridDim.y ) {
        for ( int col = blockIdx.x*blockDim.x+threadIdx.x;
                  col < w;
                  col += blockDim.x*gridDim.x ) {
            unsigned char pixval = tex2D( texImage, (float) col, (float) row );
            atomicAdd( &sHist[pixval], 1 );
        }
    }
    __syncthreads();
    unsigned int *outputHist = &tempHist[blockIdx.x][0];
    for ( int i = tid; i < 256; i += blockDim.x*blockDim.y ) {
        outputHist[i] = sHist[i];
    }
}

__global__ void
histogram1DPrivatizedPerBlockReduce(
    unsigned int *pHist,
    const unsigned char *base, size_t N )
{
    __shared__ int sHist[256];
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        sHist[i] = 0;
    }
    __syncthreads();
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        atomicAdd( &sHist[ base[i] ], 1 );
    }
    __syncthreads();
    unsigned int *outputHist = &tempHist[blockIdx.x][0];
    for ( int i = threadIdx.x; i < 256; i += blockDim.x*blockDim.y ) {
        outputHist[i] = sHist[i];
    }
}

// Must be invoked with 256 threads, one for each output histogram element
__global__ void
histogramPrivatizedPerBlockFinalReduction( 
    unsigned int *pHist, 
    unsigned int cHist )
{
    unsigned int sum = 0;
    for ( int i = 0; i < cHist; i++ ) {
        sum += tempHist[i][threadIdx.x];
    }
    pHist[threadIdx.x] = sum;
}

void
GPUhistogramPrivatizedPerBlockReduce(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads, dim3 blocks )
{
    cudaError_t status;
    cudaEvent_t start = 0, stop = 0;

    CUDART_CHECK( cudaEventCreate( &start, 0 ) );
    CUDART_CHECK( cudaEventCreate( &stop, 0 ) );

    CUDART_CHECK( cudaEventRecord( start, 0 ) );
    {
        void *ptempHist;
        CUDART_CHECK( cudaGetSymbolAddress( &ptempHist, tempHist ) );
        CUDART_CHECK( cudaMemset( ptempHist, 0, sizeof(tempHist ) ) );
    }
    //histogramPrivatizedPerBlock<<<blocks,threads>>>( pHist, x, y, w, h );
    histogram1DPrivatizedPerBlockReduce<<<240,threads.x*threads.y>>>( pHist, dptrBase, w*h );
    histogramPrivatizedPerBlockFinalReduction<<<1,256>>>( pHist, 240 );
    
    CUDART_CHECK( cudaEventRecord( stop, 0 ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}
