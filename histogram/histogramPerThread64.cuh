/*
 *
 * histogramPerThread64.cuh
 *
 * Implementation of histogram that uses 8-bit privatized counters
 * and uses 32-bit increments to process them.
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

inline __device__ void
incPrivatized32Element( unsigned char pixval )
{
    extern __shared__ unsigned int privHist[];
    const int blockDimx = 64;
    unsigned int increment = 1<<8*(pixval&3);
    int index = pixval>>2;
    privHist[index*blockDimx+threadIdx.x] += increment;
}

template<bool bClear>
__device__ void
merge64HistogramsToOutput( unsigned int *pHist )
{
    extern __shared__ unsigned int privHist[];

    unsigned int sum02 = 0;
    unsigned int sum13 = 0;
    for ( int i = 0; i < 64; i++ ) {
        int index = (i+threadIdx.x)&63;
        unsigned int myValue = privHist[threadIdx.x*64+index];
        if ( bClear ) privHist[threadIdx.x*64+index] = 0;
        sum02 += myValue & 0xff00ff;
        myValue >>= 8;
        sum13 += myValue & 0xff00ff;
    }
    
    atomicAdd( &pHist[threadIdx.x*4+0], sum02&0xffff );
    sum02 >>= 16;
    atomicAdd( &pHist[threadIdx.x*4+2], sum02 );

    atomicAdd( &pHist[threadIdx.x*4+1], sum13&0xffff );
    sum13 >>= 16;
    atomicAdd( &pHist[threadIdx.x*4+3], sum13 );
}

__global__ void
histogram1DPerThread64(
    unsigned int *pHist,
    const unsigned char *base, size_t N )
{
    extern __shared__ unsigned int privHist[];
    for ( int i = threadIdx.x;
              i < 64*blockDim.x;
              i += blockDim.x ) {
        privHist[i] = 0;
    }
    __syncthreads();
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        incPrivatized32Element( base[i] );
    }
    __syncthreads();

    merge64HistogramsToOutput<false>( pHist );
}

void
GPUhistogramPerThread64(
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
    int numblocks = INTDIVIDE_CEILING( w*h, numthreads*255 );

    cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );

    cuda(Memset( pHist, 0, 256*sizeof(unsigned int) ) );

    cuda(EventRecord( start, 0 ) );
    histogram1DPerThread64<<<numblocks,numthreads,numthreads*256>>>( pHist, dptrBase, w*h );
    cuda(EventRecord( stop, 0 ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}
