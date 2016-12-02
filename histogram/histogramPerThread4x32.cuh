/*
 *
 * histogramPerThread4x32.cuh
 *
 * This version is the same as histogramPerThread4x64.cuh,
 * but uses just one warp per block and 32 privatized histograms
 * and pads the shared memory allocation to 33 elements per row.
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
inline __device__ void
incPrivatized32Element4x33( unsigned int *pHist, unsigned int privHist[64][32], unsigned char pixval )
{
    unsigned int increment = 1<<8*(pixval&3);
    int index = pixval>>2;
    unsigned int value = privHist[index][threadIdx.x];
    value += increment;
    if ( bPeriodicMerge ) {
        if ( value & 0x80808080 ) {
            atomicAdd( pHist+pixval, 0x80 );
        }
        value &= 0x7f7f7f7f;
    }
    privHist[index][threadIdx.x] = value;
}


template<bool bClear>
__device__ void
merge32HistogramsToOutput( unsigned int *pHist, unsigned int privHist[64][32] )
{
    unsigned int sum02[2];
    unsigned int sum13[2];

    sum02[0] = sum02[1] = 0;
    sum13[0] = sum13[1] = 0;

    for ( int i = 0; i < 32; i++ ) {
        int index = (i+threadIdx.x)&0x1f;
        unsigned int myValue0 = privHist[threadIdx.x+ 0][index];
        unsigned int myValue1 = privHist[threadIdx.x+32][index];
        if ( bClear ) {
            privHist[threadIdx.x+ 0][index] = 0;
            privHist[threadIdx.x+32][index] = 0;
        }
        sum02[0] += myValue0 & 0xff00ff;
        myValue0 >>= 8;
        sum13[0] += myValue0 & 0xff00ff;

        sum02[1] += myValue1 & 0xff00ff;
        myValue1 >>= 8;
        sum13[1] += myValue1 & 0xff00ff;
        
    }
    int rowIndex = threadIdx.x;
    atomicAdd( &pHist[rowIndex*4+0], sum02[0]&0xffff );
    sum02[0] >>= 16;
    atomicAdd( &pHist[rowIndex*4+2], sum02[0] );

    atomicAdd( &pHist[rowIndex*4+1], sum13[0]&0xffff );
    sum13[0] >>= 16;
    atomicAdd( &pHist[rowIndex*4+3], sum13[0] );

    rowIndex += 32;
    atomicAdd( &pHist[rowIndex*4+0], sum02[1]&0xffff );
    sum02[1] >>= 16;
    atomicAdd( &pHist[rowIndex*4+2], sum02[1] );

    atomicAdd( &pHist[rowIndex*4+1], sum13[1]&0xffff );
    sum13[1] >>= 16;
    atomicAdd( &pHist[rowIndex*4+3], sum13[1] );

}

template<bool bPeriodicMerge>
__global__ void
histogram1DPerThread4x32(
    unsigned int *pHist,
    const unsigned char *base, size_t N )
{
    __shared__ unsigned int privHist[64][32];
    const int blockDimx = 32;

    if ( blockDim.x != blockDimx ) return;

    for ( int i = 0; i < 32; i++ ) {
        privHist[threadIdx.x][i] = 0;
        privHist[threadIdx.x+32][i] = 0;
    }
    __syncthreads();
    int cIterations = 0;
    for ( int i = blockIdx.x*blockDimx+threadIdx.x;
              i < N/4;
              i += blockDimx*gridDim.x ) {
        unsigned int value = ((unsigned int *) base)[i];
        incPrivatized32Element4x33<false>( pHist, privHist, value & 0xff ); value >>= 8;
        incPrivatized32Element4x33<false>( pHist, privHist, value & 0xff ); value >>= 8;
        incPrivatized32Element4x33<false>( pHist, privHist, value & 0xff ); value >>= 8;
        incPrivatized32Element4x33<false>( pHist, privHist, value );
        cIterations += 1;
        if ( bPeriodicMerge && cIterations>=252/4 ) {
            cIterations = 0;
            __syncthreads();
            merge32HistogramsToOutput<true>( pHist, privHist );
        }
    }
    __syncthreads();

    merge32HistogramsToOutput<false>( pHist, privHist );
}

template<bool bPeriodicMerge>
void
GPUhistogramPerThread4x32(
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
    histogram1DPerThread4x32<bPeriodicMerge><<<numblocks,32>>>( pHist, dptrBase, w*h );
    cuda(EventRecord( stop, 0 ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( ms, start, stop ) );
Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return;
}

void
GPUhistogramPerThread4x32(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerThread4x32<false>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}

void
GPUhistogramPerThread4x32_PeriodicMerge(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    GPUhistogramPerThread4x32<true>( ms, pHist, dptrBase, dPitch, x, y, w, h, threads );
}
