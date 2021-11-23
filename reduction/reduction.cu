/*
 *
 * Microbenchmark for reduction (summation) of 32-bit integers.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_11 <options> reduction.cu
 * Requires: SM 1.1 for global atomics.
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
#include <chCommandLine.h>
#include <chError.h>

#include "reduction1ExplicitLoop.cuh"
#include "reduction2WarpSynchronous.cuh"
#include "reduction3WarpSynchronousTemplated.cuh"
#include "reduction4SinglePass.cuh"
#include "reduction5Atomics.cuh"
#include "reduction6AnyBlockSize.cuh"
#include "reduction7SyncThreads.cuh"

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
    int *answer, const int *deviceIn, size_t N, 
    int cBlocks, int cThreads,
    pfnReduction hostReduction
)
{
    double ret = 0.0;
    int *deviceAnswer = 0;
    int *partialSums = 0;
    cudaEvent_t start = 0;
    cudaEvent_t stop = 0;
    cudaError_t status;

    cuda(Malloc( &deviceAnswer, sizeof(int) ) );
    cuda(Malloc( &partialSums, cBlocks*sizeof(int) ) );
    cuda(EventCreate( &start ) );
    cuda(EventCreate( &stop ) );
    cuda(DeviceSynchronize() );

    cuda(EventRecord( start, 0 ) );
    hostReduction( 
        deviceAnswer, 
        partialSums, 
        deviceIn, 
        N, 
        cBlocks, 
        cThreads );
    cuda(EventRecord( stop, 0 ) );
    cuda(Memcpy( 
        answer, 
        deviceAnswer, 
        sizeof(int), 
        cudaMemcpyDeviceToHost ) );

    ret = chEventBandwidth( start, stop, N*sizeof(int) ) / 
        powf(2.0f,30.0f);

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
    for ( int cThreads = 128; cThreads <= props.maxThreadsPerBlock; cThreads*=2 ) {
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

    cuda(Malloc( &smallArray, N*sizeof(int) ) );
    cuda(Malloc( &partialSums, 1*sizeof(int) ) );
    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        pfnReduction( partialSums, partialSums, smallArray, N, 1, 256 );
    }
    cuda(DeviceSynchronize() );
    chTimerGetTime( &stop );
    ret = chTimerElapsedTime( &start, &stop );
    ret = (ret / (double) cIterations) * 1e6;
Error:
    (void) cudaFree( partialSums );
    (void) cudaFree( smallArray );
    return ret;
}

void
Usage()
{
    printf( "Command-line options\n" );
    printf( "    --device <deviceID>: specify device to run the test on\n" );
    printf( "    --n <N>: specify number of Mintegers to process (scaled by 1048576)\n" );
    printf( "    --throughput: compute throughput of different reduction implementations\n" );
    printf( "    --help or --usage: generate this message\n" );
}

int
main( int argc, char *argv[] )
{
    cudaDeviceProp props;
    cudaError_t status;
    int *hostData = 0;
    int *deviceData = 0;
    int sum;
    int cMInts = 32;
    size_t cInts;
    int device = 0;

    if ( chCommandLineGetBool( "usage", argc, argv ) ) {
        Usage();
        return 0;
    }

    chCommandLineGet( &cMInts, "n", argc, argv );
    cInts = cMInts * 1048576;
    chCommandLineGet( &device, "device", argc, argv );

    hostData = (int *) malloc( cInts*sizeof(int) );
    if ( ! hostData )
        goto Error;
    cuda(SetDevice( device ) );
    cuda(SetDeviceFlags( cudaDeviceMapHost ) );
    cuda(Malloc( &deviceData, cInts*sizeof(int) ) );
    cuda(GetDeviceProperties( &props, 0 ) );

    sum = 0;
    for ( size_t i = 0; i < cInts; i++ ) {
        int value = rand()&1;
        sum += value;
        hostData[i] = value;
    }
    cuda(Memcpy( deviceData, hostData, cInts*sizeof(int), 
        cudaMemcpyHostToDevice ) );

    {
    
        struct {
            const char *szName;
            pfnReduction pfn;
        } rgTests[] = { { "simple loop", Reduction1 },
                        { "warpsync", Reduction2 },
                        { "templated", Reduction3 },
                        { "single pass", Reduction4 },
                        { "global atomic", Reduction5 },
                        { "any block size", Reduction6 },
                        { "syncthreads", Reduction7 }
                       };

        const size_t numTests = sizeof(rgTests)/sizeof(rgTests[0]);
        TimingResult result[numTests];

        if ( chCommandLineGetBool( "throughput", argc, argv ) ) {
            printf( "Microseconds per reduction operation:\n" );
            for ( size_t i = 0; i < numTests; i++ ) {
                printf( "\t%s: %.2f\n", rgTests[i].szName, 
                    usPerInvocation( 100000, 1, rgTests[i].pfn ) );
            }
            exit(0);
        }

        sum = 0;
        for ( size_t i = 0; i < cInts; i++ ) {
            sum += hostData[i];
        }

        printf( "Testing on %d integers\n", cInts );
        printf( "\t\t" );
        for ( int i = 128; i <= props.maxThreadsPerBlock; i *= 2 ) {
            printf( "%d\t", i );
        }
        printf( "maxThr\tmaxBW\n" );
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
