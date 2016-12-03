/*
 *
 * radixSort.cu
 *
 * Microdemo and microbenchmark of Radix Sort.  CPU only for now.
 *
 * Build with: nvcc -I ../chLib <options> radixSort.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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


#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

#include <chTimer.h>
#include <chError.h>


#define NUM_THREADS 64

template<const int b>
__global__ void
RadixHistogram_device( int *dptrHistogram, const int *in, size_t N, int shift, int mask )
{
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        int index = (in[i] & mask) >> shift;
        atomicAdd( dptrHistogram+index, 1 );
    }
#if 0
    const int cBuckets = 1<<b;
    __shared__ unsigned char sharedHistogram[NUM_THREADS][cBuckets];

    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        int index = (in[i] & mask) >> shift;
        if ( 0 == ++sharedHistogram[threadIdx.x][index] ) {
            atomicAdd( dptrHistogram+index, 256 );
        }
    }
    __syncthreads();
    for ( int i = 0; i < cBuckets; i++ ) {
        if ( sharedHistogram[threadIdx.x][i] ) {
            atomicAdd( dptrHistogram+i, sharedHistogram[threadIdx.x][i] );
        }
    }
#endif
}

template<const int b>
void
RadixHistogram( int *dptrHistogram, const int *in, size_t N, int shift, int mask, int cBlocks, int cThreads )
{
    RadixHistogram_device<b><<<cBlocks,cThreads>>>( dptrHistogram, in, N, shift, mask );
}


template<const int b>
bool
RadixPass( int *out, const int *in, size_t N, int shift, int mask )
{
    bool ret = false;
    cudaError_t status;
    const int numCounts = 1<<b;
    int counts[numCounts];
    memset( counts, 0, sizeof(counts) );

int *gpuIn = 0;
int *gpuHistogram = 0;
int *cpuHistogram = 0;
cuda(Malloc( &gpuIn, N*sizeof(int) ) );
cuda(Memcpy( gpuIn, in, N*sizeof(int), cudaMemcpyHostToDevice ) );
cuda(Malloc( &gpuHistogram, (1<<b)*sizeof(int) ) );
cuda(Memset( gpuHistogram, 0, (1<<b)*sizeof(int) ) );
cpuHistogram = (int *) malloc( (1<<b)*sizeof(int) );
if ( ! cpuHistogram ) {
    status = cudaErrorMemoryAllocation;
    goto Error;
}

RadixHistogram<b>( gpuHistogram, gpuIn, N, shift, mask, 1500, 512 );
cuda(Memcpy( cpuHistogram, gpuHistogram, (1<<b)*sizeof(int), cudaMemcpyDeviceToHost ) );


    for ( size_t i = 0; i < N; i++ ) {
        int value = in[i];
        int index = (value & mask) >> shift;
        counts[index] += 1;
    }

for ( int j = 0; j < (1<<b); j++ ) {
    if ( counts[j] != cpuHistogram[j] )
        __debugbreak();
}

    //
    // compute exclusive scan of counts
    //
    {
        int sum = 0;
        for ( int i = 0; i < numCounts; i++ ) { 
            int temp = counts[i];
            counts[i] = sum;
            sum += temp;
        }
    }

    //
    // scatter each input to the correct output
    //
    for ( size_t i = 0; i < N; i++ ) {
        int value = in[i];
        int index = (value & mask) >> shift;
        out[ counts[index] ] = value;
        counts[index] += 1;
    }
    ret = true;
Error:
    cudaFree( gpuIn );
    cudaFree( gpuHistogram );
    free( cpuHistogram );
    
    return ret;
}

template<const int b>
int *
RadixSort( int *out[2], const int *in, size_t N )
{
    int shift = 0;
    int mask = (1<<b)-1;

    //
    // index of output array, ping-pongs between 0 and 1.
    //
    int outIndex = 0;

    RadixPass<b>( out[outIndex], in, N, shift, mask );
    while ( mask ) {
        outIndex = 1 - outIndex;
        shift += 1;
        mask <<= 1;
        RadixPass<b>( out[outIndex], out[1-outIndex], N, shift, mask );
    }
    return out[outIndex];

}

bool
TestSort( float *et, int *(*pfnSort)( int *[2], const int *, size_t ), size_t N, int mask = -1 )
{
    chTimerTimestamp start, stop;
    bool ret = false;
    int *sortInput = new int[ N ];
    int *sortOutput[2];
    int *radixSortedArray = 0;
    std::vector<int> sortedOutput( N );
    sortOutput[0] = new int[ N ];
    sortOutput[1] = new int[ N ];

    if ( 0 == sortInput || 
         0 == sortOutput[0] ||
         0 == sortOutput[1] ) {
        goto Error;
    }

    for ( int i = 0; i < N; i++ ) {
        sortedOutput[i] = sortInput[i] = (rand()|(rand()<<16)) & mask;
    }

    {
        std::sort( sortedOutput.begin(), sortedOutput.end() );
    }

    chTimerGetTime( &start );

    //
    // RadixSort returns sortOutput[0] or sortOutput[1],
    // depending on where it wound up in the ping-pong
    // between output arrays.
    //
    radixSortedArray = pfnSort( sortOutput, sortInput, N );

    chTimerGetTime( &stop );
    *et = chTimerElapsedTime( &start, &stop );

    for ( size_t i = 0; i < N; i++ ) {
        if ( radixSortedArray[i] != sortedOutput[i] ) {
#ifdef _WIN32
            __debugbreak();
#endif
            goto Error;
        }
    }
    ret = true;
Error:
    delete[] sortInput;
    delete[] sortOutput[0];
    delete[] sortOutput[1];
    return ret;
}

int
main()
{
    float ms;
    size_t N = 16*1048576;

#define TEST_VECTOR( fn, N, mask )  \
    if ( ! TestSort( &ms, fn, N, mask ) ) {  \
        printf( "%s (N=%d, mask=%d) FAILED\n", #fn, (int) N, mask );  \
        exit(1);    \
    } \
    else { \
        printf( "%s (N=%d, mask=%d): %.2f Melements/s\n", #fn, (int) N, mask, (double) (N/1e6)/(ms/1000.0f) ); \
    }

//    TEST_VECTOR( 32, 0xf );

    TEST_VECTOR( RadixSort<1>, N, 0xffffffff );
    TEST_VECTOR( RadixSort<2>, N, 0xffffffff );
    TEST_VECTOR( RadixSort<4>, N, 0xffffffff );

    TEST_VECTOR( RadixSort<1>, N, 0xf );
    TEST_VECTOR( RadixSort<2>, N, 0xf );
    TEST_VECTOR( RadixSort<4>, N, 0xf );

    TEST_VECTOR( RadixSort<1>, N, 0x1 );
    TEST_VECTOR( RadixSort<2>, N, 0x1 );
    TEST_VECTOR( RadixSort<4>, N, 0x1 );

}
