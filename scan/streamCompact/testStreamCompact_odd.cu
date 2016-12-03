/*
 *
 * testStreamCompact_odd.cu
 *
 * Microdemo to test a special case of stream compaction
 * (extracting odd integers from an input array).
 *
 * Build with: nvcc -I ..\chLib <options> testStreamCompact_odd.cu
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
#include <chUtil.h>

#include "scanWarp.cuh"
#include "scanBlock.cuh"

#include "scanZeroPad.cuh"

#define min(a,b) ((a)<(b)?(a):(b))

#include "streamCompact_odd.cuh"

template<class T>
bool
TestStreamCompact( 
    const char *szScanFunction, 
    void (*pfnScanGPU)(T *, int *, const T *, size_t, int), 
    size_t N, 
    int numThreads,
    float fRatio )
{
    bool ret = false;
    cudaError_t status;

    T *inGPU = 0;
    T *outGPU = 0;
    int *hostTotal = 0;
    int *deviceTotal = 0;
    T *inCPU = (T *) malloc( N*sizeof(T) );
    T *outCPU = (T *) malloc( N*sizeof(T) );
    T *hostGPU = (T *) malloc( N*sizeof(T) );
    int *pScanCPU = (int *) malloc( N*sizeof(int) );
    if ( 0 == inCPU || 0==outCPU || 0==hostGPU || 0==pScanCPU )
        goto Error;

    printf( "Testing %s (%d integers, %d threads/block)\n", 
        szScanFunction,
        (int) N,
        numThreads );

    cuda(HostAlloc( &hostTotal, sizeof(int), cudaHostAllocMapped ) );
    cuda(HostGetDevicePointer( (void **) &deviceTotal, hostTotal, 0 ) );

    cuda(Malloc( &inGPU, N*sizeof(T) ) );
    cuda(Malloc( &outGPU, N*sizeof(T) ) );
    cuda(Memset( inGPU, 0, N*sizeof(T) ) );
    cuda(Memset( outGPU, 0, N*sizeof(T) ) );

    srand(0);
    for ( int i = 0; i < N; i++ ) {
        inCPU[i] = rand() & ~1; // random even number
        // about fRatio*N values are odd
        if ( (float) rand() / (float) RAND_MAX <= fRatio ) {
            inCPU[i] |= 1;
        }
    }

    cuda(Memcpy( inGPU, inCPU, N*sizeof(T), cudaMemcpyHostToDevice ) );
    pfnScanGPU( outGPU, deviceTotal, inGPU, N, numThreads );
    cuda(Memcpy( hostGPU, outGPU, N*sizeof(T), cudaMemcpyDeviceToHost ) );
    {
        size_t inxOut = 0;
        size_t inxIn = 0;
        while ( inxIn < N && inxOut < *hostTotal ) {
            if ( isOdd( inCPU[inxIn] ) ) {
                if ( hostGPU[inxOut] != inCPU[inxIn] ) {
                    printf( "Scan failed\n" );
                    goto Error;
                }
                inxOut += 1;
            }
            inxIn += 1;
        }
        while ( inxIn < N ) {
            if ( isOdd( inCPU[inxIn] ) ) {
                printf( "Missed some inputs that met the criteria\n" );
            }
            inxIn += 1;
        }
        if ( inxOut != *hostTotal ) {
            printf( "Total reported is incorrect\n" );
        }
    }
    ret = true;
Error:
    cudaFree( outGPU );
    cudaFree( inGPU );
    cudaFreeHost( hostTotal );
    free( inCPU );
    free( outCPU );
    free( hostGPU );
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

    printf( "Testing WITHOUT zero padding:\n" );
    for ( float fRatio = 0.25f; fRatio <= 1.0f; fRatio *= 2.0f ) {
        for ( int numThreads = 128; numThreads <= maxThreads; numThreads *= 2 ) {
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, false>, 32, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, false>, 1024, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, false>, 1020, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, false>, 16*1024*1024, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, false>, 16*1024*1024 -10, numThreads, fRatio );
        }
    }

    printf( "Testing WITH zero padding:\n" );
    for ( float fRatio = 0.25f; fRatio <= 1.0f; fRatio *= 2.0f ) {
        for ( int numThreads = 128; numThreads <= maxThreads; numThreads *= 2 ) {
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, true>, 32, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, true>, 1024, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, true>, 1020, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, true>, 16*1024*1024, numThreads, fRatio );
            TestStreamCompact<int>( "streamCompact_odd", streamCompact_odd<int, true>, 16*1024*1024 -10, numThreads, fRatio );
        }
    }
}
