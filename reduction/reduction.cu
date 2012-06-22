/*
 *
 * Microbenchmark for reduction (summation) of 32-bit integers.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_12 <options> reduction.cu
 * Requires: SM 1.2 for shared atomics (as well as global atomics).
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

#include <stdio.h>
#include <stdlib.h>

#include <assert.h>

#include <chTimer.h>
#include <chError.h>

#include "reduction1ExplicitLoop.cu"
#include "reduction2WarpSynchronous.cu"
#include "reduction3WarpSynchronousTemplated.cu"
#include "reduction4SinglePass.cu"
#include "reduction5GlobalAtomics.cu"
#include "reduction6SharedAtomics.cu"

typedef struct TimingResult_struct {
    double Bandwidth;
    int numThreads;
} TimingResult;

double
chEventBandwidth( cudaEvent_t start, cudaEvent_t stop, double cBytes )
{
    float ms;
    if ( cudaSuccess != cudaEventElapsedTime( &ms, start, stop ) )
        return 0.0;
    return cBytes * 1000.0f / ms;
}

typedef void (*pfnReduction)(int *out, int *intermediateSums, const int *in, size_t N, int cBlocks, int cThreads);

double
TimedReduction( 
    int *answer, const int *deviceIn, size_t N, int cBlocks, int cThreads,
    pfnReduction hostReduction
)
{
    double ret = 0.0;
    int *deviceAnswer = 0;
    int *partialSums = 0;
    cudaEvent_t start = 0;
    cudaEvent_t stop = 0;
    cudaError_t status;

    CUDART_CHECK( cudaMalloc( &deviceAnswer, sizeof(int) ) );
    CUDART_CHECK( cudaMalloc( &partialSums, cBlocks*sizeof(int) ) );
    CUDART_CHECK( cudaEventCreate( &start ) );
    CUDART_CHECK( cudaEventCreate( &stop ) );
    CUDART_CHECK( cudaThreadSynchronize() );

    CUDART_CHECK( cudaEventRecord( start, 0 ) );
    hostReduction( deviceAnswer, partialSums, deviceIn, N, cBlocks, cThreads );
    CUDART_CHECK( cudaEventRecord( stop, 0 ) );
    CUDART_CHECK( cudaMemcpy( answer, deviceAnswer, sizeof(int), cudaMemcpyDeviceToHost ) );

    ret = chEventBandwidth( start, stop, N*sizeof(int) ) / 1073741824.0;

    // fall through to free resources before returning
Error:
    cudaFree( deviceAnswer );
    cudaFree( partialSums );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return ret;
}

void
Shmoo( TimingResult *timingResult,
       int *deviceData, size_t cInts, int expectedSum, 
       bool bPrint, bool bPrintMax,
       void (*pfnReduce)(int *out, int *intermediateSums, const int *in, size_t N, int cBlocks, int cThreads) )
{
    double maxBW = 0.0f;
    int maxThreads;
    int cBlocks = 1800;
    cudaDeviceProp props;

    cudaGetDeviceProperties( &props, 0 );
    for ( int cThreads = 128; cThreads <= 512; cThreads*=2 ) {
        int sum = 0;
        double bw = TimedReduction( &sum, deviceData, cInts, cBlocks, cThreads, pfnReduce );
        if( sum != expectedSum ) {
            printf( "Sum is wrong: 0x%08x should be 0x%08x\n", sum, expectedSum );
            exit(1);
        }
        if ( bPrint ) {
            printf( "%.2f\t", bw );
        }
        if ( bw > maxBW ) {
            maxBW = bw;
            maxThreads = cThreads;
        }
    }
    timingResult->numThreads = maxThreads;
    timingResult->Bandwidth = maxBW;
    if ( bPrintMax) {
        printf( "Max bandwidth of %.2f G/s attained by %d blocks "
            "of %d threads\n", maxBW, cBlocks, maxThreads );
    }
}

double
usPerInvocation( int cIterations, size_t N,
    void (*pfnReduction)( int *out, int *partial, const int *in, size_t N, int numBlocks, int numThreads ) )
{
    cudaError_t status;
    int *smallArray = 0;
    int *partialSums = 0;
    double ret = 0.0f;
    chTimerTimestamp start, stop;

    CUDART_CHECK( cudaMalloc( &smallArray, N*sizeof(int) ) );
    CUDART_CHECK( cudaMalloc( &partialSums, 1*sizeof(int) ) );
    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        pfnReduction( partialSums, partialSums, smallArray, N, 1, 256 );
    }
    CUDART_CHECK( cudaThreadSynchronize() );
    chTimerGetTime( &stop );
    ret = chTimerElapsedTime( &start, &stop );
    ret = (ret / (double) cIterations) * 1e6;
Error:
    (void) cudaFree( partialSums );
    (void) cudaFree( smallArray );
    return ret;
}

int
main( int argc, char *argv[] )
{
    cudaDeviceProp props;
    cudaError_t status;
    int *hostData = 0;
    int *deviceData = 0;
    int sum;
    size_t cInts;

    if ( argc != 2 ) {
        printf( "Usage: %s <N> where <N> is the number of Mints to allocate\n", argv[0] );
        exit(1);
    }
    cInts = (size_t) atoi(argv[1]) * 1048576;

    hostData = (int *) malloc( cInts*sizeof(int) );
    if ( ! hostData )
        goto Error;
    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    CUDART_CHECK( cudaMalloc( &deviceData, cInts*sizeof(int) ) );
    CUDART_CHECK( cudaGetDeviceProperties( &props, 0 ) );

    sum = 0;
    for ( size_t i = 0; i < cInts; i++ ) {
        int value = rand();
        sum += value;
        hostData[i] = value;
    }
    CUDART_CHECK( cudaMemcpy( deviceData, hostData, cInts*sizeof(int), 
        cudaMemcpyHostToDevice ) );

#if 1
    {
        printf( "Microseconds per reduction operation:\n" );
        const size_t N = 1;
        printf( "\tReduction1: %.2f\n", usPerInvocation( 100000, N, Reduction1 ) );
        printf( "\tReduction2: %.2f\n", usPerInvocation( 100000, N, Reduction2 ) );
        printf( "\tReduction3: %.2f\n", usPerInvocation( 100000, N, Reduction3 ) );
        printf( "\tReduction4: %.2f\n", usPerInvocation( 100000, N, Reduction4 ) );
        printf( "\tReduction5: %.2f\n", usPerInvocation( 100000, N, Reduction5 ) );
        printf( "\tReduction6: %.2f\n", usPerInvocation( 100000, N, Reduction6 ) );

exit(0);
    
    }
#endif
    
    {
        struct {
            const char *szName;
            pfnReduction pfn;
        } rgTests[] = { { "explicit loop", Reduction1 },
                        { "warpsync", Reduction2 },
                        { "templated", Reduction3 },
                        { "single pass", Reduction4 },
                        { "global atomic", Reduction5 },
                        { "shared atomic", Reduction6 },
                       };

        const size_t numTests = sizeof(rgTests)/sizeof(rgTests[0]);
        TimingResult result[numTests];

        sum = 0;
        for ( size_t i = 0; i < cInts; i++ ) {
            sum += hostData[i];
        }

        printf( "Testing on %d integers\n", cInts );
        printf( "\t\t128\t256\t512\tmaxThr\tmaxBW\n" );
        for ( size_t i = 0; i < numTests; i++ ) {
            printf( "%s\t", rgTests[i].szName );
            Shmoo( &result[i], deviceData, cInts, sum, true, false, rgTests[i].pfn );
            printf( "%d\t%.2f\n", result[i].numThreads, result[i].Bandwidth );
        }
        printf( "\n" );
    }

    return 0;
Error:
    free( hostData );
    if ( deviceData ) {
        cudaFree( deviceData );
    }
    return 1;
}
