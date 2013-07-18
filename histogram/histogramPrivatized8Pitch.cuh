/*
 *
 * histogramPrivatized8.cuh
 *
 * Implementation of histogram that uses a separate histogram per
 * thread, and accumulates overflows by firing global atomics.
 *
 * Requires: SM 1.1, for global atomics.
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

__device__ inline
void incHistElement( unsigned int *pHist, unsigned char *myHistogram, int pixval )
{
    myHistogram[pixval]++;
    if ( ! myHistogram[pixval] ) {
        atomicAdd( pHist+pixval, 256 );
    }
}


__global__ void
histogramPrivatized8Pitch( 
    unsigned int *pHist, 
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y, 
    int w, int h )
{
    extern __shared__ unsigned char privateHistogram[];
    const int tid = threadIdx.y*blockDim.x+threadIdx.x;
    unsigned char *myHistogram = privateHistogram+256*tid;

    for ( int i = tid; i < 256*blockDim.x*blockDim.y/4; i += blockDim.x*blockDim.y ) {
        ((unsigned int *) privateHistogram)[i] = 0;
    }
    __syncthreads();

    for ( int row = blockIdx.y*blockDim.y+threadIdx.y; 
              row < h;
              row += blockDim.y*gridDim.y ) {
        unsigned int *pRow = (unsigned int *) (dptrBase+row*dPitch);
        for ( int col = blockIdx.x*blockDim.x+threadIdx.x;
                  col < w/4;
                  col += blockDim.x*gridDim.x ) {
            unsigned int pixval4 = pRow[col];
            incHistElement( pHist, myHistogram, pixval4 & 0xff ); pixval4 >>= 8;
            incHistElement( pHist, myHistogram, pixval4 & 0xff ); pixval4 >>= 8;
            incHistElement( pHist, myHistogram, pixval4 & 0xff ); pixval4 >>= 8;
            incHistElement( pHist, myHistogram, pixval4 );
        }
    }
    __syncthreads();
    for ( int iHistogram = 0; iHistogram < blockDim.x*blockDim.y; iHistogram++ ) {
        unsigned char *myHistogram = privateHistogram+256*iHistogram;
        for ( int i = tid; i < 256/4; i += blockDim.x*blockDim.y ) {
            unsigned int myHistVal = ((unsigned int *) myHistogram)[i];
            {
                if ( 0xff&myHistVal ) atomicAdd( pHist+i*4+0, myHistVal & 0xff );
                    myHistVal >>= 8;
                if ( 0xff&myHistVal ) atomicAdd( pHist+i*4+1, myHistVal & 0xff );
                    myHistVal >>= 8;
                if ( 0xff&myHistVal ) atomicAdd( pHist+i*4+2, myHistVal & 0xff );
                    myHistVal >>= 8;
                if (      myHistVal ) atomicAdd( pHist+i*4+3, myHistVal );
            }
        }
    }
}

void
GPUhistogramPrivatized8Pitch(
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
    histogramPrivatized8Pitch<<<blocks,threads, threads.x*threads.y*256>>>( pHist, dptrBase, dPitch, x, y, w, h );
    CUDART_CHECK( cudaEventRecord( stop, 0 ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( ms, start, stop ) );
    
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}
