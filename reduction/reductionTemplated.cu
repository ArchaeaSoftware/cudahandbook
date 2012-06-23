/*
 *
 * reductionTemplated.cu
 *
 * Microbenchmark/demo that uses templates to efficiently compute
 * reductions for any data type.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_11 <options> reductionTemplated.cu
 * Requires: SM 1.1, for global atomics.
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
#include <stdarg.h>

#include <assert.h>

#include <chCommandLine.h>
#include <chTimer.h>
#include <chError.h>

void
forkPrint( FILE *out, const char *fmt, ... )
{
    va_list arg;
    va_start(arg, fmt);
    vprintf( fmt, arg );
    if ( out ) {
        vfprintf( out, fmt, arg );
    }
    va_end(arg);
}

FILE *g_fileBlocks;
FILE *g_fileShmoo;

//
// From Reduction SDK sample:
//
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
//
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*) (void *) __smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*) (void *) __smem;
    }
};

#include "reduction_Sumi.h"
#include "reduction_Sumi_isq.h"

#include "reduction_Sumf.h"
#include "reduction_Sumf_fsq.h"

#include "reduction1Templated.cuh"
#include "reduction2Templated.cuh"
#include "reduction3Templated.cuh"
#include "reduction4Templated.cuh"

typedef struct TimingResult_struct {
    double us;
    int fastestBlocks;
    int fastestThreads;
} TimingResult;

double
chEventBandwidth( cudaEvent_t start, cudaEvent_t stop, double cBytes )
{
    float ms;
    if ( cudaSuccess != cudaEventElapsedTime( &ms, start, stop ) )
        return 0.0;
    return cBytes * 1000.0f / ms;
}

typedef void (*pfnReduction)(CReduction_Sumi *out, CReduction_Sumi *partialSums, const int *in, size_t N, int cBlocks, int cThreads);

template<class ReductionType, class T>
double
TimedReduction( 
    ReductionType *answer, const T *deviceIn, size_t N, int cBlocks, int cThreads,
    void (*hostReduction)(ReductionType *out, ReductionType *partialSums, const T *in, size_t N, int cBlocks, int cThreads)
)
{
    double ret = 0.0;
    ReductionType *deviceAnswer = 0;
    ReductionType *partialSums = 0;
    float ms;
    cudaEvent_t start = 0;
    cudaEvent_t stop = 0;
    cudaError_t status;

    CUDART_CHECK( cudaMalloc( &deviceAnswer, sizeof(ReductionType) ) );
    CUDART_CHECK( cudaMalloc( &partialSums, cBlocks*sizeof(ReductionType) ) );
    CUDART_CHECK( cudaEventCreate( &start ) );
    CUDART_CHECK( cudaEventCreate( &stop ) );
    CUDART_CHECK( cudaThreadSynchronize() );

    CUDART_CHECK( cudaEventRecord( start, 0 ) );
    hostReduction( deviceAnswer, partialSums, deviceIn, N, cBlocks, cThreads );
    CUDART_CHECK( cudaEventRecord( stop, 0 ) );
    CUDART_CHECK( cudaMemcpy( answer, deviceAnswer, sizeof(T), cudaMemcpyDeviceToHost ) );

    CUDART_CHECK( cudaEventElapsedTime( &ms, start, stop ) )
    ret = ms * 1000.0f;

    // fall through to free resources before returning
Error:
    cudaFree( deviceAnswer );
    cudaFree( partialSums );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return ret;
}

int g_ShmooBlockMin = 2;
int g_ShmooBlockMax = 900;
int g_ShmooBlockStep = 1;
int g_ShmooThreadMin = 32;
int g_ShmooThreadMax = 512;

size_t g_nMin;
size_t g_nMax;
size_t g_nStep;

template<class ReductionType, class T>
void
Shmoo( TimingResult *timingResult,
       T *deviceData, size_t cInts, const ReductionType& expectedSum, 
       bool bPrint, 
       void (*pfnReduce)(ReductionType *out, ReductionType *partialSums, 
           const T *in, size_t N, int cBlocks, int cThreads) )
{
    double minTime = 1e6;
    int fastestThreads = 0;
    int fastestBlocks = 0;
    if ( bPrint ) {
        forkPrint( g_fileBlocks, "Blocks\t" );
        for ( int cThreads  = g_ShmooThreadMin; 
                  cThreads <= g_ShmooThreadMax; 
                  cThreads *= 2 ) {
            forkPrint( g_fileBlocks, "%d\t", cThreads );
        }
        forkPrint( g_fileBlocks, "minTC\tminTime\n" );        
    }
    for ( int cBlocks  = g_ShmooBlockMin; 
              cBlocks <= g_ShmooBlockMax; 
              cBlocks += g_ShmooBlockStep ) {
        int fastestThreadsThisBlock = 0;
        double minTimeThisBlock = 1e6;
        if ( bPrint ) {
            forkPrint( g_fileBlocks, "%d\t", cBlocks );
        }
        for ( int  cThreads = g_ShmooThreadMin; 
                  cThreads <= g_ShmooThreadMax; 
                  cThreads *= 2 ) {
            ReductionType sum;
            double us = TimedReduction( &sum, deviceData, cInts, cBlocks, cThreads, pfnReduce );
            if( sum != expectedSum ) {
                printf( "Sum is wrong: 0x%08x should be 0x%08x\n", sum.sum, expectedSum.sum );
                exit(1);
            }
            if ( us < minTimeThisBlock ) {
                minTimeThisBlock = us;
                fastestThreadsThisBlock = cThreads;
            }
            if ( us < minTime ) {
                minTime = us;
                fastestBlocks = cBlocks;
                fastestThreads = cThreads;
            }
            if ( bPrint ) {
                forkPrint( g_fileBlocks, "%.2f\t", us );
                if ( cThreads == 512 ) {
                    forkPrint( g_fileBlocks, "%d\t%.2f\n", fastestThreadsThisBlock, minTimeThisBlock );
                }
            }
            
        }
    }
    timingResult->fastestBlocks = fastestBlocks;
    timingResult->fastestThreads = fastestThreads;
    timingResult->us = minTime;
}

template<class ReductionType, class T>
double
usPerInvocation( int cIterations, size_t N,
    void (*pfnReduction)( ReductionType *out, ReductionType *partial, 
        const T *in, size_t N, int numBlocks, int numThreads ) )
{
    cudaError_t status;
    T *smallArray = 0;
    ReductionType *partialSums = 0;
    double ret = 0.0f;
    chTimerTimestamp start, stop;

    CUDART_CHECK( cudaMalloc( &smallArray, N*sizeof(T) ) );
    CUDART_CHECK( cudaMalloc( &partialSums, 1*sizeof(ReductionType) ) );
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

template<class ReductionType, class T>
void
ReportOverhead()
{
    printf( "Microseconds per reduction operation:\n" );
    const size_t N = 1;
    printf( "\tReduction1: %.2f\n", usPerInvocation<ReductionType, T>( 100000, N, Reduction1 ) );
    printf( "\tReduction2: %.2f\n", usPerInvocation<ReductionType, T>( 100000, N, Reduction2 ) );
    printf( "\tReduction3: %.2f\n", usPerInvocation<ReductionType, T>( 100000, N, Reduction3 ) );
    printf( "\tReduction4: %.2f\n", usPerInvocation<ReductionType, T>( 100000, N, Reduction4 ) );
}

template<class ReductionType, class T>
bool
ShmooReport( size_t nMin, size_t nMax, size_t nStep, bool bFloat, bool bReportBlockSizes )
{
    cudaError_t status;
    cudaDeviceProp props;
    bool ret = false;
    T *hostData = 0;
    T *deviceData = 0;

    forkPrint( g_fileShmoo, "Shmoo: %d %ss (start at %d, step by %d)\n", (int) nMax,
        (bFloat) ? "float" : "int", (int) nMin, (int) nStep );
    hostData = (T *) malloc( nMax*sizeof(T) );
    if ( ! hostData )
        goto Error;
    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    CUDART_CHECK( cudaMalloc( &deviceData, nMax*sizeof(T) ) );
    CUDART_CHECK( cudaGetDeviceProperties( &props, 0 ) );
    CUDART_CHECK( cudaMemcpy( deviceData, hostData, nMax*sizeof(T), cudaMemcpyHostToDevice ) );

    if ( ! bReportBlockSizes ) {
        forkPrint( g_fileShmoo, "N\tBlocks(1)\tThreads(1)\tus(1)\t"
                                   "Blocks(2)\tThreads(2)\tus(2)\t"
                                   "Blocks(3)\tThreads(3)\tus(3)\t"
                                   "Blocks(4)\tThreads(4)\tus(4)\n" );
    }

    for ( size_t count = nMin; count <= nMax; count += nStep ) {
        TimingResult result[4];

        void (*rgpfnReduction[4])(ReductionType *out, 
            ReductionType *partialSums, 
            const T *in, size_t N, int cBlocks, int cThreads) =
            { Reduction1<ReductionType, T>, 
              Reduction2<ReductionType, T>,
              Reduction3<ReductionType, T>,
              Reduction4<ReductionType, T> };
        ReductionType sum;

        if ( ! bReportBlockSizes ) {
            forkPrint( g_fileShmoo, "%d\t", count );
        }

        for ( size_t i = 0; i < count; i++ ) {
            sum += hostData[i];
        }

        for ( size_t i = 0; i < 4; i++ ) {
            if ( bReportBlockSizes ) {
                forkPrint( g_fileShmoo, "Formulation %d, %d: reporting for all block sizes\n", i, count );
            }
            Shmoo( &result[i], deviceData, count, sum, bReportBlockSizes, rgpfnReduction[i] );
            if ( ! bReportBlockSizes ) {
                forkPrint( g_fileShmoo, "%d\t%d\t%.2f\t", result[i].fastestBlocks, 
                    result[i].fastestThreads, result[i].us );
            }
            fflush(stdout);
        }
        forkPrint( g_fileShmoo, "\n" );
    }
Error:
    free( hostData );
    cudaFree( deviceData );
    return ret;
}

void
Usage()
{
    printf( "Usage: reductionTemplated [options]\n" );
    printf( "    Options may include any of the following:\n" );
    printf( "        --device <device>] - specify device to run on\n" );
    printf( "        --float - perform analysis on float-valued reductions\n" );
    printf( "        --squares - perform analysis on sum+sum of squares reductions (i.e. more computation)\n" );
    printf( "        --overhead - instead of performing shmoo over range of input sizes, report kernel launch overhead\n" );
    printf( "        --nMax <nMax> - number of elements (will be multiplied by 1024) to run analysis on\n" );
    printf( "        --nMin <nMin> - start shmoo at this number of elements (scaled by 1024)\n" );
    printf( "        --nStep <nStep> - step number of elements by this amount (scaled by 1024) until it reaches nMax\n" );

    printf( "        --blockMax <nMax> - maximum number of blocks to run analysis on\n" );
    printf( "        --blockMin <nMin> - minimum number of blocks to run analysis on\n" );
    printf( "        --blockStep <nStep> - step number of blocks to run analysis on\n" );

    printf( "        --threadMax <nMax> - maximum number of threads to run analysis on\n" );
    printf( "        --threadMin <nMin> - minimum number of threads to run analysis on\n" );

    printf( "        --blocksFile <filename> - name of file to which to dump blocks reporting\n" );
    printf( "        --shmooFile <filename> - name of file to which to dump shmoo reporting\n" );

}

int
main( int argc, char *argv[] )
{
    int device = 0;
    int nMini, nMaxi, nStepi;
    size_t nMin, nMax, nStep;
    bool bDoFloats = false;
    bool bDoSquares = false;
    bool bDoOverhead = false;
    bool bReportBlockSizes = false;

    if ( argc == 1 ) {
        Usage();
        exit(0);
    }

    if ( chCommandLineGet( &device, "device", argc, argv ) ) {
        printf( "Using device %d...\n", device );
        cudaSetDevice( device );
    }

    // By default, analyze just 1M elements, 256K at a time.
    nMini = 256;
    nStepi = 256;
    nMaxi = 1024;
    chCommandLineGet( &nMini, "nMin", argc, argv );
    chCommandLineGet( &nStepi, "nStep", argc, argv );
    chCommandLineGet( &nMaxi, "nMax", argc, argv );
    if ( nMini > nMaxi ) {
        fprintf( stderr, "nMax must be greater than or equal to nMin\n" );
        exit(1);
    }
    chCommandLineGet( &g_ShmooBlockMin, "blockMin", argc, argv );
    chCommandLineGet( &g_ShmooBlockMax, "blockMax", argc, argv );
    if ( g_ShmooBlockMin > g_ShmooBlockMax ) {
        fprintf( stderr, "blockMax must be greater than or equal to blockMin\n" );
        exit(1);
    }
    chCommandLineGet( &g_ShmooBlockStep, "blockStep", argc, argv );
    chCommandLineGet( &g_ShmooThreadMin, "threadMin", argc, argv );
    chCommandLineGet( &g_ShmooThreadMax, "threadMax", argc, argv );
    if ( g_ShmooThreadMin < 32 || g_ShmooThreadMax < 32 || 
         g_ShmooThreadMax > 512 || g_ShmooThreadMin > 512 ) {
        fprintf( stderr, "threadMin/threadMax must be in range 32..512\n" );
        exit(1);
    }
    if ( g_ShmooBlockMin > g_ShmooBlockMax ) {
        fprintf( stderr, "threadMax must be greater than or equal to threadMin\n" );
        exit(1);
    }
    if ( g_ShmooThreadMin > g_ShmooThreadMax ) {
        fprintf( stderr, "threadMax must be greater than or equal to threadMin\n" );
        exit(1);
    }
    if ( (g_ShmooThreadMin & (g_ShmooThreadMin-1)) || 
         (g_ShmooThreadMax & (g_ShmooThreadMax-1)) ) {
        fprintf( stderr, "threadMin and threadMax must be powers of 2\n" );
        exit(1);
    }
    nMin = (size_t) nMini * 1024;
    nMax = (size_t) nMaxi * 1024;
    nStep = (size_t) nStepi * 1024;
    bDoFloats = chCommandLineGetBool( "float", argc, argv );
    bDoSquares = chCommandLineGetBool( "squares", argc, argv );
    bDoOverhead = chCommandLineGetBool( "overhead", argc, argv );
    bReportBlockSizes = chCommandLineGetBool( "report-block-sizes", argc, argv );

    printf( "Threads: %d - %d\n", g_ShmooThreadMin, g_ShmooThreadMax );
    printf( "Threads: %d - %d by %d\n", g_ShmooBlockMin, g_ShmooBlockMax, g_ShmooBlockStep );

    {
        char *fname;
        if ( chCommandLineGet( &fname, "blocksFile", argc, argv ) ) {
            g_fileBlocks = fopen( fname, "wt" );
            bReportBlockSizes = g_fileBlocks != NULL;
        }
        if ( chCommandLineGet( &fname, "shmooFile", argc, argv ) ) {
            g_fileShmoo = fopen( fname, "wt" );
        }
    }

    if ( bDoOverhead ) {
        if ( bDoSquares ) {
            ReportOverhead<CReduction_Sumi_isq, int>();
        }
        else {
            ReportOverhead<CReduction_Sumi, int>();
        }
        exit(0);
    }

    if ( bDoFloats ) {
        if ( bDoSquares ) {
            ShmooReport<CReduction_Sumf_fsq, float>( nMin, nMax, nStep, true, bReportBlockSizes );
        }
        else {
            ShmooReport<CReduction_Sumf, float>( nMin, nMax, nStep, true, bReportBlockSizes );
        }
    }
    else {
        if ( bDoSquares ) {
            ShmooReport<CReduction_Sumi_isq, int>( nMin, nMax, nStep, false, bReportBlockSizes );
        }
        else {
            ShmooReport<CReduction_Sumi, int>( nMin, nMax, nStep, false, bReportBlockSizes );
        }
    }
    if ( g_fileShmoo ) {
        fclose( g_fileShmoo );
    }
    if ( g_fileBlocks ) {
        fclose( g_fileBlocks );
    }
}
