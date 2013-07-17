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

__global__ void
histogramPrivatized8( 
    unsigned char *privateHistograms,
    unsigned int *pHist, 
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
        for ( int col = blockIdx.x*blockDim.x+threadIdx.x;
                  col < w;
                  col += blockDim.x*gridDim.x ) {
            unsigned char pixval = tex2D( texImage, (float) col, (float) row );
            myHistogram[pixval]++;
            if ( ! myHistogram[pixval] ) {
                atomicAdd( pHist+pixval, 256 );
            }
        }
    }
    __syncthreads();
    for ( int iHistogram = 0; iHistogram < blockDim.x*blockDim.y; iHistogram++ ) {
        unsigned char *myHistogram = privateHistogram+256*iHistogram;
//        unsigned char *myOutHistogram = privateHistograms+256*iHistogram;
        for ( int i = tid; i < 256/4; i += blockDim.x*blockDim.y ) {
//            myOutHistogram[i] = myHistogram[i];
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
//            unsigned char myHistVal = myHistogram[i];
//            if ( myHistVal ) atomicAdd( pHist+i, myHistVal );
        }
    }
}

void
GPUhistogramPrivatized8(
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads, dim3 blocks )
{
    unsigned char *hptrPrivateHistograms;
    unsigned char *dptrPrivateHistograms;
    cudaError_t status;
    
    CUDART_CHECK( cudaHostAlloc( &hptrPrivateHistograms, threads.x*threads.y*256, cudaHostAllocMapped ) );
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrPrivateHistograms, hptrPrivateHistograms, 0 ) );
    
    histogramPrivatized8<<<blocks,threads, threads.x*threads.y*256>>>( dptrPrivateHistograms, pHist, x, y, w, h );
    CUDART_CHECK( cudaDeviceSynchronize() );
    
Error:
    return;
}
