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

template<typename ReductionType, typename T>
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

    cuda(Malloc( &deviceAnswer, sizeof(ReductionType) ) );
    cuda(Malloc( &partialSums, cBlocks*sizeof(ReductionType) ) );
    cuda(EventCreate( &start ) );
    cuda(EventCreate( &stop ) );
    cuda(DeviceSynchronize() );

    cuda(EventRecord( start, 0 ) );
    hostReduction( deviceAnswer, partialSums, deviceIn, N, cBlocks, cThreads );
    cuda(EventRecord( stop, 0 ) );
    cuda(Memcpy( answer, deviceAnswer, sizeof(T), cudaMemcpyDeviceToHost ) );

    cuda(EventElapsedTime( &ms, start, stop ) )
    ret = ms * 1000.0f;

    // fall through to free resources before returning
Error:
    cudaFree( deviceAnswer );
    cudaFree( partialSums );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return ret;
}

int g_cBlocks = 900;

int g_ShmooThreadMin = 32;
int g_ShmooThreadMax = 512;

template<typename ReductionType, typename T>
void
Shmoo( TimingResult *timingResult,
       T *deviceData, size_t cInts, const ReductionType& expectedSum, 
       void (*pfnReduce)(ReductionType *out, ReductionType *partialSums, 
           const T *in, size_t N, int cBlocks, int cThreads) )
{
    double minTime = 1e6;
    int fastestThreads = 0;
    for ( int cThreads  = g_ShmooThreadMin; 
              cThreads <= g_ShmooThreadMax; 
              cThreads *= 2 ) {
        printf( "%d\t", cThreads );
    }
    printf( "minTC\tminTime\n" );        
    for ( int  cThreads = g_ShmooThreadMin; 
              cThreads <= g_ShmooThreadMax; 
              cThreads *= 2 ) {
        ReductionType sum;
        double us = TimedReduction( &sum, deviceData, cInts, g_cBlocks, cThreads, pfnReduce );
        if( sum != expectedSum ) {
            printf( "Sum is wrong: 0x%08x should be 0x%08x\n", sum.sum, expectedSum.sum );
            exit(1);
        }
        if ( us < minTime ) {
            minTime = us;
            fastestThreads = cThreads;
        }
        printf( "%.2f\t", us );
        if ( cThreads == g_ShmooThreadMax ) {
            printf( "%d\t%.2f\n", fastestThreads, minTime );
        }
        
    }
    timingResult->fastestThreads = fastestThreads;
    timingResult->us = minTime;
}

template<typename ReductionType, typename T>
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

    cuda(Malloc( &smallArray, N*sizeof(T) ) );
    cuda(Malloc( &partialSums, 1*sizeof(ReductionType) ) );
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

template<typename ReductionType, typename T>
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

template<typename ReductionType, typename T>
bool
ShmooReport( size_t N, bool bFloat )
{
    cudaError_t status;
    cudaDeviceProp props;
    bool ret = false;
    T *hostData = 0;
    T *deviceData = 0;

    forkPrint( g_fileShmoo, "Shmoo: %d %ss\n", (int) N, 
        (bFloat) ? "float" : "int" );
    hostData = (T *) malloc( N*sizeof(T) );
    if ( ! hostData )
        goto Error;
    cuda(SetDeviceFlags( cudaDeviceMapHost ) );
    cuda(Malloc( &deviceData, N*sizeof(T) ) );
    cuda(GetDeviceProperties( &props, 0 ) );
    cuda(Memcpy( deviceData, hostData, N*sizeof(T), cudaMemcpyHostToDevice ) );

    forkPrint( g_fileShmoo, "N\tThreads(1)\tus(1)\t"
                               "Threads(2)\tus(2)\t"
                               "Threads(3)\tus(3)\t"
                               "Threads(4)\tus(4)\n" );

    {
        TimingResult result[4];

        void (*rgpfnReduction[4])(ReductionType *out, 
            ReductionType *partialSums, 
            const T *in, size_t N, int cBlocks, int cThreads) =
            { Reduction1<ReductionType, T>, 
              Reduction2<ReductionType, T>,
              Reduction3<ReductionType, T>,
              Reduction4<ReductionType, T> };
        ReductionType sum;

        forkPrint( g_fileShmoo, "%d\t", N );

        for ( size_t i = 0; i < N; i++ ) {
            sum += hostData[i];
        }

        for ( size_t i = 0; i < 4; i++ ) {
            Shmoo( &result[i], deviceData, N, sum, rgpfnReduction[i] );
            forkPrint( g_fileShmoo, "%d\t%.2f\t",
                result[i].fastestThreads, result[i].us );
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
    printf( "        --size <N> - number of Melements (scaled by 1048576)\n" );

    printf( "        --blocks <N> - number of blocks to run analysis on\n" );

    printf( "        --threadMax <count> - maximum number of threads to run analysis on\n" );
    printf( "        --threadMin <count> - minimum number of threads to run analysis on\n" );

    printf( "        --shmooFile <filename> - name of file to which to dump shmoo reporting\n" );

}

int
main( int argc, char *argv[] )
{
    int device = 0;
    int iN = 16;
    size_t N;
    bool bDoFloats = false;
    bool bDoSquares = false;
    bool bDoOverhead = false;

    if ( argc == 1 ) {
        Usage();
        exit(0);
    }

    if ( chCommandLineGet( &device, "device", argc, argv ) ) {
        printf( "Using device %d...\n", device );
        cudaSetDevice( device );
    }

    chCommandLineGet( &iN, "size", argc, argv );
    chCommandLineGet( &g_cBlocks, "blocks", argc, argv );
    chCommandLineGet( &g_ShmooThreadMin, "threadMin", argc, argv );
    chCommandLineGet( &g_ShmooThreadMax, "threadMax", argc, argv );
    if ( g_ShmooThreadMin < 32 || g_ShmooThreadMax < 32 || 
         g_ShmooThreadMax > 512 || g_ShmooThreadMin > 512 ) {
        fprintf( stderr, "threadMin/threadMax must be in range 32..512\n" );
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
    
    N = (size_t) iN * 1048576;
    bDoFloats = chCommandLineGetBool( "float", argc, argv );
    bDoSquares = chCommandLineGetBool( "squares", argc, argv );
    bDoOverhead = chCommandLineGetBool( "overhead", argc, argv );

    printf( "Threads: %d - %d\n", g_ShmooThreadMin, g_ShmooThreadMax );
    printf( "Blocks: %d\n", g_cBlocks );

    {
        char *fname;
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
            ShmooReport<CReduction_Sumf_fsq, float>( N, true );
        }
        else {
            ShmooReport<CReduction_Sumf, float>( N, true );
        }
    }
    else {
        if ( bDoSquares ) {
            ShmooReport<CReduction_Sumi_isq, int>( N, false );
        }
        else {
            ShmooReport<CReduction_Sumi, int>( N, false );
        }
    }
    if ( g_fileShmoo ) {
        fclose( g_fileShmoo );
    }
}
