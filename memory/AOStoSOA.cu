/*
 *
 * AOStoSOA.cu
 *
 * Microdemo that illustrates how to convert from AOS (array of
 * structures) to SOA (structure of arrays) representation.
 *
 * Build with: nvcc -I ../chLib <options> AOStoSOA.cu
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

#include <stdio.h>
#include <stdlib.h>

#include <chError.h>

#include "AOStoSOA_1.cuh"
#include "AOStoSOA_2.cuh"

template<typename T, const int k>
double
TestAOStoSOA( 
    size_t N, 
    void (*pfnAOStoSOA)( T **out, const T *in, size_t N, int cBlocks, int cThreads ),
    int cIterations = 1 )
{
    double ret = 0.0;
    cudaError_t status;
    T **dptrpSOA = 0;
    T *dptrSOA[k];
    T *dptrAOS = 0;
    T *hrefSOA[k];  // host reference
    T *hrefAOS = 0;

    cudaEvent_t evStart = 0;
    cudaEvent_t evStop = 0;

    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );

    memset( hrefSOA, 0, sizeof(hrefSOA) );
    memset( dptrSOA, 0, sizeof(dptrSOA) );
    CUDART_CHECK( cudaMalloc( &dptrAOS, N*k*sizeof(T) ) );
    CUDART_CHECK( cudaHostAlloc( &hrefAOS, N*k*sizeof(T), cudaHostAllocMapped ) );

    CUDART_CHECK( cudaMalloc( &dptrpSOA, k*sizeof(T *) ) );
    for ( int i = 0; i < k; i++ ) {
        CUDART_CHECK( cudaHostAlloc( &hrefSOA[i], N*sizeof(T), cudaHostAllocMapped ) );
        memset( hrefSOA[i], 0, N*sizeof(T) );
        CUDART_CHECK( cudaMalloc( &dptrSOA[i], N*sizeof(T) ) );
        CUDART_CHECK( cudaMemset( dptrSOA[i], 0, N*sizeof(T) ) );
    }
    CUDART_CHECK( cudaMemcpy( dptrpSOA, dptrSOA, k*sizeof(T *), cudaMemcpyHostToDevice ) );

    for ( size_t i = 0; i < N; i++ ) {
        for ( int j = 0; j < k; j++ ) {
            hrefAOS[i*k+j] = j<<24|i;//rand();
        }
    }
    CUDART_CHECK( cudaMemcpyAsync( dptrAOS, hrefAOS, N*k*sizeof(T), cudaMemcpyHostToDevice ) );
    pfnAOStoSOA( dptrpSOA, dptrAOS, N, 1500, 512 );
    for ( int i = 0; i < k; i++ ) {
        CUDART_CHECK( cudaMemcpyAsync( hrefSOA[i], dptrSOA[i], N*sizeof(T), cudaMemcpyDeviceToHost ) );
    }
    CUDART_CHECK( cudaDeviceSynchronize() );
    for ( int i = 0; i < N; i++ ) {
        for ( int j = 0; j < k; j++ ) {
            if ( hrefAOS[i*k+j] != hrefSOA[j][i] ) {
                printf( "Mismatch at i==%d, k==%d (%d should be %d)\n", i, j, hrefSOA[j][i], hrefAOS[i*k+j] );
                goto Error;
            }
        }
    }

    CUDART_CHECK( cudaEventRecord( evStart, NULL ) );
    for ( int i = 0; i < cIterations; i++ ) {
        pfnAOStoSOA( dptrpSOA, dptrAOS, N, 1500, 512 );
    }
    CUDART_CHECK( cudaEventRecord( evStop, NULL ) );
    CUDART_CHECK( cudaDeviceSynchronize() );

    {
        float ms;
        CUDART_CHECK( cudaEventElapsedTime( &ms, evStart, evStop ) );
        ret = (double) N*cIterations*sizeof(T)*1000.0 / ms;
    }

Error:
    cudaEventDestroy( evStart );
    cudaEventDestroy( evStop );
    cudaFreeHost( hrefAOS );
    cudaFree( dptrAOS );
    cudaFree( dptrpSOA );
    for ( int i = 0; i < k; i++ ) {
        cudaFreeHost( hrefSOA[i] );
        cudaFree( dptrSOA[i] );
    }
    return ret;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    int iN = 32;
    cudaError_t status;

    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    CUDART_CHECK( cudaFree(0) );

    #define TEST_VECTOR(fn) { \
        double bytesPerSecond = TestAOStoSOA<int, 3>( 1048576*iN, fn<int,3>, 10 ); \
        if ( 0.0 == bytesPerSecond ) \
            goto Error; \
        printf( "%s: %.2f Gbytes/s\n", #fn, bytesPerSecond/1e9 ); \
    }

    TEST_VECTOR( AOStoSOA_1 );
    TEST_VECTOR( AOStoSOA_2 );

    ret = 0;
Error:
    return ret;
}
