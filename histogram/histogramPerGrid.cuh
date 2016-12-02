/*
 *
 * histogramPerGrid.cuh
 *
 * Implementation of histogram that uses one global atomic per pixel.
 * This results in very data-dependent performance, as the hardware
 * facilities for mutual exclusion contend when trying to increment
 * the same histogram value concurrently.
 *
 * Requires: SM 1.1, for global atomics.
 *
 * Build with:
 *    nvcc -I ..\chLib --gpu-architecture sm_xx histogram.cu ..\chLib\pgm.cu -lnpp
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

__global__ void
histogram1DPerGrid(
    unsigned int *pHist,
    const unsigned char *base, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x+threadIdx.x;
                 i < N/4;
                 i += blockDim.x*gridDim.x ) {
        unsigned int value = ((unsigned int *) base)[i];
        atomicAdd( &pHist[ value & 0xff ], 1 ); value >>= 8;
        atomicAdd( &pHist[ value & 0xff ], 1 ); value >>= 8;
        atomicAdd( &pHist[ value & 0xff ], 1 ); value >>= 8;
        atomicAdd( &pHist[ value ]       , 1 );
    }
}

void
GPUhistogramPerGrid(
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
//    histogramPerGrid<<<blocks,threads>>>( pHist, w, h );
    histogram1DPerGrid<<<400, 256>>>( pHist, dptrBase, w*h );
    cuda(EventRecord( stop, 0 ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}
