/*
 *
 * testScan.cu
 *
 * Microdemo to test scan algorithms.
 *
 * Build with: nvcc -I ..\chLib <options> testScan.cu
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

#include <chTimer.h>
#include <chAssert.h>
#include <chError.h>

#include "scanWarp.cuh"
#include "scanBlock.cuh"

#include "scanZeroPad.cuh"

#define min(a,b) ((a)<(b)?(a):(b))

enum ScanType {
    Inclusive, Exclusive
};

#include "scanFan.cuh"
#include "scanReduceThenScan.cuh"
#include "scanReduceThenScan_0.cuh"
#include "scan2Level.cuh"
#include "scanThrust.cuh"

void
ScanExclusiveCPU( int *out, const int *in, size_t N )
{
    int sum = 0;
    for ( size_t i = 0; i < N; i++ ) {
        int next = in[i]; // in case we are doing this in place
        out[i] = sum;
        sum += next;
    }
}

int
ScanInclusiveCPU( int *out, const int *in, size_t N )
{
    int sum = 0;
    for ( size_t i = 0; i < N; i++ ) {
        sum += in[i];
        out[i] = sum;
    }
    return sum;
}

void
RandomArray( int *out, size_t N, int modulus )
{
    for ( size_t i = 0; i < N; i++ ) {
        out[i] = rand() % modulus;
    }
}

template<class T>
bool
TestScan( const char *szScanFunction, 
          void (*pfnScanGPU)(T *, const T *, size_t, int), 
          size_t N, 
          int numThreads )
{
    bool ret = false;
    cudaError_t status;
    int *inGPU = 0;
    int *outGPU = 0;
    int *inCPU = (T *) malloc( N*sizeof(T) );
    int *outCPU = (int *) malloc( N*sizeof(T) );
    int *hostGPU = (int *) malloc( N*sizeof(T) );
    if ( 0==inCPU || 0==outCPU || 0==hostGPU )
        goto Error;

    printf( "Testing %s (%d integers, %d threads/block)\n", 
        szScanFunction,
        (int) N,
        numThreads );

    cuda(Malloc( &inGPU, N*sizeof(T) ) );
    cuda(Malloc( &outGPU, N*sizeof(T) ) );
    cuda(Memset( inGPU, 0, N*sizeof(T) ) );
    cuda(Memset( outGPU, 0, N*sizeof(T) ) );

    cuda(Memset( outGPU, 0, N*sizeof(T) ) );

    RandomArray( inCPU, N, 256 );
for ( int i = 0; i < N; i++ ) {
    inCPU[i] = i;
}
    
    ScanInclusiveCPU( outCPU, inCPU, N );

    cuda(Memcpy( inGPU, inCPU, N*sizeof(T), cudaMemcpyHostToDevice ) );
    pfnScanGPU( outGPU, inGPU, N, numThreads );
    cuda(Memcpy( hostGPU, outGPU, N*sizeof(T), cudaMemcpyDeviceToHost ) );
    for ( size_t i = 0; i < N; i++ ) {
        if ( hostGPU[i] != outCPU[i] ) {
            printf( "Scan failed\n" );
#ifdef _WIN32
            __debugbreak();
#else
            assert(0);
#endif
            goto Error;
        }
    }
    ret = true;
Error:
    cudaFree( outGPU );
    cudaFree( inGPU );
    free( inCPU );
    free( outCPU );
    free( hostGPU );
    return ret;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int maxThreads;

    cuda(SetDevice( 0 ) );
    cuda(SetDeviceFlags( cudaDeviceMapHost ) );

    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, 0 );
        maxThreads = prop.maxThreadsPerBlock;
    }

#define SCAN_TEST_VECTOR( Function, N, numThreads ) do { \
    srand(0); \
    bool bSuccess = TestScan<int>( #Function, Function, N, numThreads ); \
    if ( ! bSuccess ) { \
        printf( "%s failed: N=%d, numThreads=%d\n", #Function, N, numThreads ); \
        exit(1); \
    } \
} while (0)

    for ( int numThreads = 256; numThreads <= maxThreads; numThreads *= 2 ) {
        
        for ( int numInts = 256; numInts <= 2048; numInts += 128 ) {

            SCAN_TEST_VECTOR( scan2Level<int>, numInts, numThreads );

            SCAN_TEST_VECTOR( scanFan<int>, numInts, numThreads );
            SCAN_TEST_VECTOR( scanReduceThenScan<int>, numInts, numThreads );
            SCAN_TEST_VECTOR( scanReduceThenScan_0<int>, numInts, numThreads );
            SCAN_TEST_VECTOR( scan2Level<int>, numInts, numThreads );
            SCAN_TEST_VECTOR( scan2Level_0<int>, numInts, numThreads );
        }

        for ( int numInts = 33*1048576-1; numInts < 33*1048576+1; numInts++ ) {

            SCAN_TEST_VECTOR( scan2Level<int>, numInts, numThreads );
            SCAN_TEST_VECTOR( scan2Level_0<int>, numInts, numThreads );

            SCAN_TEST_VECTOR( scanFan<int>, numInts, numThreads );
            SCAN_TEST_VECTOR( scanReduceThenScan<int>, numInts, numThreads );
            SCAN_TEST_VECTOR( scanReduceThenScan_0<int>, numInts, numThreads );

            SCAN_TEST_VECTOR( ScanThrust<int>, numInts, numThreads );
        }

    }
    return 0;
Error:
    return 1;
}
