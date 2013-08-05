/*
 *
 * histogramSharedAtomic.cuh
 *
 * Implementation of histogram that uses one shared atomic per pixel.
 * This results in very data-dependent performance, as the hardware
 * facilities for mutual exclusion contend when trying to increment
 * the same histogram value concurrently.
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
histogramSharedAtomic( 
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

static __device__ unsigned int
value_to_index( unsigned int i, unsigned int offset ) 
{
    return i;//return (i+offset)&0xff;
}

template<bool bUnroll>
__global__ void
histogram1DSharedAtomic(
    unsigned int *pHist,
    const unsigned char *base, size_t N, unsigned int offset )
{
    __shared__ int sHist[257];
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

            atomicAdd( &sHist[ value_to_index( value & 0xff, offset ) ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value_to_index( value & 0xff, offset ) ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value_to_index( value & 0xff, offset ) ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value_to_index( value, offset ) ]       , 1 );
        }
        else {
            atomicAdd( &sHist[ base[i] ], 1 );
        }
    }
    __syncthreads();
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        atomicAdd( &pHist[i], sHist[ value_to_index(i, offset) ] );
    }
      
}

template<bool bUnroll>
void
GPUhistogramSharedAtomic(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads, dim3 blocks, unsigned int offset=0 )
{
    cudaError_t status;
    cudaEvent_t start = 0, stop = 0;

    CUDART_CHECK( cudaEventCreate( &start, 0 ) );
    CUDART_CHECK( cudaEventCreate( &stop, 0 ) );

    CUDART_CHECK( cudaEventRecord( start, 0 ) );
    //histogramSharedAtomic<<<blocks,threads>>>( pHist, x, y, w, h );
    histogram1DSharedAtomic<bUnroll><<<400,256/*threads.x*threads.y*/>>>( pHist, dptrBase, w*h, offset );
    CUDART_CHECK( cudaEventRecord( stop, 0 ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}


void
GPUhistogramSharedAtomic(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads, dim3 blocks )
{
    GPUhistogramSharedAtomic<false>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads, blocks, 1 );
}

void
GPUhistogramSharedAtomic4x(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads, dim3 blocks )
{
    GPUhistogramSharedAtomic<true>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads, blocks, 1 );
}

