/*
 *
 * timeStreamCompact_odd.cu
 *
 * Microbenchmark to time a special case of stream compaction
 * (extracting odd integers from an input array).
 *
 * Build with: nvcc -I ..\chLib <options> timeStreamCompact_odd.cu
 * Requires: No minimum SM requirement.
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

#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <chTimer.h>
#include <chAssert.h>
#include <chError.h>

#include "scanWarp.cuh"
#include "scanBlock.cuh"

#include "scanZeroPad.cuh"

#define min(a,b) ((a)<(b)?(a):(b))

int *g_hostIn, *g_hostOut;

#include "streamCompact_odd.cuh"

template <class T>
struct is_odd : public thrust::unary_function<T,bool>
{
    __host__ __device__
    bool operator()(T x)
    {
        return x & 1;
    }
};

template<class T, bool bZeroPad>
void
streamCompact_odd_Thrust( T *out, int *outCount, const T *in, size_t N, int b )
{
    thrust::device_ptr<T> thrustOut(out);
    thrust::device_ptr<T> thrustOutCount(outCount);
    thrust::device_ptr<T> thrustIn((T *) in);
    thrust::copy_if( thrustIn, thrustIn+N, thrustOut, is_odd<int>() );
}


template<class T>
double
TimeStreamCompact( 
    const char *szScanFunction,
    void (*pfnScanGPU)(T *, int *, const T *, size_t, int), 
    size_t N, 
    int numThreads, 
    int cIterations,
    float fRatio )  // ratio of numbers to be odd
{
    chTimerTimestamp start, stop;
    cudaError_t status;

    double ret = 0.0;

    int *inGPU = 0;
    int *outGPU = 0;
    int *hostTotal = 0;
    int *deviceTotal = 0;
    int *inCPU = (int *) malloc( N*sizeof(T) );
    int *outCPU = (int *) malloc( N*sizeof(T) );
    if ( 0==inCPU || 0==outCPU )
        goto Error;
    cuda(HostAlloc( &hostTotal, sizeof(int), cudaHostAllocMapped ) );
    cuda(HostGetDevicePointer( (void **) &deviceTotal, hostTotal, 0 ) );

    cuda(Malloc( &inGPU, N*sizeof(T) ) );
    cuda(Malloc( &outGPU, N*sizeof(T) ) );

    srand(0);
    for ( int i = 0; i < N; i++ ) {
        inCPU[i] = rand() & ~1; // random number with bit 0 clear
        // At least fRatio of the input samples must be odd
        if ( (float) rand() / (float) RAND_MAX <= fRatio ) {
            inCPU[i] |= 1;
        }
    }

    cuda(Memcpy( inGPU, inCPU, N*sizeof(T), cudaMemcpyHostToDevice ) );
    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        pfnScanGPU( outGPU, deviceTotal, inGPU, N, numThreads );
    }
    if ( cudaSuccess != cudaDeviceSynchronize() )
        goto Error;
    chTimerGetTime( &stop );

    // ints per second
    ret = (double) cIterations*N / chTimerElapsedTime( &start, &stop );

    printf( "%s (%d threads/block): %.2f Gints/s (ratio: %.2f)\n", szScanFunction, numThreads, ret/1e9, fRatio );

Error:
    cudaFree( outGPU );
    cudaFree( inGPU );
    free( inCPU );
    free( outCPU );
    return ret;
}

int
main( int argc, char *argv[] )
{
    int maxThreads;

    cudaSetDevice( 0 );
    cudaSetDeviceFlags( cudaDeviceMapHost );
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, 0 );
        maxThreads = prop.maxThreadsPerBlock;
    }

    printf( "Timing results WITHOUT zero padding:\n" );
        for ( float fRatio = 0.25f; fRatio <= 1.0f; fRatio *= 2.0f ) {
            for ( int numThreads = 128; numThreads <= maxThreads; numThreads *= 2 ) {
                TimeStreamCompact<int>( "streamCompact_odd", 
                                        streamCompact_odd<int, false>, 
                                        32*1024*1024, 
                                        numThreads, 
                                        10, 
                                        fRatio );
            }
        }
        for ( float fRatio = 0.25f; fRatio <= 1.0f; fRatio *= 2.0f ) {
            TimeStreamCompact<int>( "streamCompact_odd (Thrust)", 
                                    streamCompact_odd_Thrust<int, false>, 
                                    32*1024*1024, 
                                    0, 
                                    10, 
                                    fRatio );
        }
    printf( "\nTiming results WITH zero padding:\n" );
        for ( float fRatio = 0.25f; fRatio <= 1.0f; fRatio *= 2.0f ) {
            for ( int numThreads = 128; numThreads <= maxThreads; numThreads *= 2 ) {
                TimeStreamCompact<int>( "streamCompact_odd", 
                                        streamCompact_odd<int, true>, 
                                        32*1024*1024, 
                                        numThreads, 
                                        10, 
                                        fRatio );
            }
        }
        for ( float fRatio = 0.25f; fRatio <= 1.0f; fRatio *= 2.0f ) {
            TimeStreamCompact<int>( "streamCompact_odd (Thrust)", 
                                    streamCompact_odd_Thrust<int, true>, 
                                    32*1024*1024, 
                                    0, 
                                    10, 
                                    fRatio );
        }
}
