/*
 *
 * histogramPerBlock.cuh
 *
 * Implementation of histogram that uses one shared atomic per pixel.
 * This results in very data-dependent performance, as the hardware
 * facilities for mutual exclusion contend when trying to increment
 * the same histogram value concurrently.
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

template<bool bUnroll>
__global__ void
histogram1DPerBlock(
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
    if ( bUnroll ) {
        N /= 4;
    }
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        if ( bUnroll ) {
            unsigned int value = ((unsigned int *) base)[i];

            atomicAdd( &sHist[ value & 0xff ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value & 0xff ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value & 0xff ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value ]       , 1 );
        }
        else {
            atomicAdd( &sHist[ base[i] ], 1 );
        }
    }
    __syncthreads();
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        atomicAdd( &pHist[i], sHist[ i ] );
    }
      
}

template<bool bUnroll>
void
GPUhistogramPerBlock(
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
    //histogramPerBlock<<<blocks,threads>>>( pHist, x, y, w, h );
    histogram1DPerBlock<bUnroll><<<400,256/*threads.x*threads.y*/>>>( pHist, dptrBase, w*h );
    cuda(EventRecord( stop, 0 ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}


void
GPUhistogramPerBlock(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerBlock<false>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}

void
GPUhistogramPerBlock4x(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerBlock<true>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}

