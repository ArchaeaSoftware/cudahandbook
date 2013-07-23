/*
 *
 * histogramSharedPrivatized.cuh
 *
 * Implementation of histogram that uses 8-bit privatized counters
 * and periodically accumulates those into a block-wide histogram
 * in shared memory.
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

__global__ void
histogramSharedPrivatized( 
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
    for ( int i = tid; i < 256; i += blockDim.x*blockDim.y ) {
        int value = sHist[i];
        if ( value ) {
            atomicAdd( &pHist[i], value );
        }
    }
}

__global__ void
histogram1DSharedPrivatized(
    unsigned int *pHist,
    const unsigned char *base, size_t N )
{
    __shared__ int sHist[256];
    extern __shared__ unsigned char privatizedHist[];
    unsigned char *myHist = privatizedHist+256*threadIdx.x;
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        sHist[i] = 0;
    }
    for ( int i = 0; i < 256; i++ ) myHist[i] = 0;
    __syncthreads();
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        unsigned char val = ++myHist[ base[i] ];
        if ( 0==val ) {
            atomicAdd( &sHist[val], 256 );
        }
    }
    __syncthreads();
    for ( int i = 0; i < 256; i++ ) {
        unsigned char val = myHist[i];
        if ( val ) atomicAdd( &sHist[i], val );
    }
    __syncthreads();
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        atomicAdd( &pHist[i], sHist[i] );
    }
      
}

void
GPUhistogramSharedPrivatized(
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
    //histogramSharedPrivatized<<<blocks,threads>>>( pHist, x, y, w, h );
    histogram1DSharedPrivatized<<<400,threads.x*threads.y,threads.x*threads.y*256>>>( pHist, dptrBase, w*h );
    CUDART_CHECK( cudaEventRecord( stop, 0 ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}
