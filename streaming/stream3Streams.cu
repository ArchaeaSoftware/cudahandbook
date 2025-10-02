/*
 *
 * stream3Streams.cu
 *
 * Formulation of stream2Async.cu that uses streams to overlap data
 * transfers and kernel processing.
 *
 * Build with: nvcc -I ../chLib stream3Streams.cu
 *
 * Copyright (c) 2012, Archaea Software, LLC.
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

#include <chError.h>
#include <chCommandLine.h>
#include <chTimer.h>

#include <stdio.h>
#include <stdlib.h>

#include "saxpyCPU.h"
#include "saxpyGPU.cuh"

cudaError_t
MeasureTimes( 
    float *msTotal,
    float *msWallClock,
    size_t N, 
    float alpha,
    int nStreams,
    int nBlocks, 
    int nThreads )
{
    cudaError_t status;
    chTimerTimestamp chStart, chStop;
    float *dptrOut = 0, *hptrOut = 0;
    float *dptrY = 0, *hptrY = 0;
    float *dptrX = 0, *hptrX = 0;
    cudaStream_t *streams = 0;
    cudaEvent_t evStart = 0;
    cudaEvent_t evStop = 0;
    size_t streamStep = N / nStreams;

    if ( N % nStreams ) {
        printf( "Stream count must be evenly divisible into N\n" );
        status = cudaErrorInvalidValue;
        goto Error;
    }

    streams = new cudaStream_t[nStreams];
    if ( ! streams ) {
        status = cudaErrorMemoryAllocation;
        goto Error;
    }
    memset( streams, 0, nStreams*sizeof(cudaStream_t) );
    for ( int i = 0; i < nStreams; i++ ) {
        cuda(StreamCreate( &streams[i] ) );
    }
    cuda(HostAlloc( &hptrOut, N*sizeof(float), 0 ) );
    memset( hptrOut, 0, N*sizeof(float) );
    cuda(HostAlloc( &hptrY, N*sizeof(float), 0 ) );
    cuda(HostAlloc( &hptrX, N*sizeof(float), 0 ) );

    cuda(Malloc( &dptrOut, N*sizeof(float) ) );
    cuda(Memset( dptrOut, 0, N*sizeof(float) ) );

    cuda(Malloc( &dptrY, N*sizeof(float) ) );
    cuda(Memset( dptrY, 0, N*sizeof(float) ) );

    cuda(Malloc( &dptrX, N*sizeof(float) ) );
    cuda(Memset( dptrY, 0, N*sizeof(float) ) );

    cuda(EventCreate( &evStart ) );
    cuda(EventCreate( &evStop ) );

    chTimerGetTime( &chStart );

    cuda(EventRecord( evStart, 0 ) );

    for ( int iStream = 0; iStream < nStreams; iStream++ ) {
        cuda(MemcpyAsync( 
                          dptrX+iStream*streamStep, 
                          hptrX+iStream*streamStep, 
                          streamStep*sizeof(float), 
                          cudaMemcpyHostToDevice, 
                          streams[iStream] ) );
        cuda(MemcpyAsync( 
                          dptrY+iStream*streamStep, 
                          hptrY+iStream*streamStep, 
                          streamStep*sizeof(float), 
                          cudaMemcpyHostToDevice, 
                          streams[iStream] ) );
    }

    for ( int iStream = 0; iStream < nStreams; iStream++ ) {
        saxpyGPU<<<nBlocks, nThreads, 0, streams[iStream]>>>( 
            dptrOut+iStream*streamStep, 
            dptrX+iStream*streamStep, 
            dptrY+iStream*streamStep, 
            streamStep, 
            alpha );
    }

    for ( int iStream = 0; iStream < nStreams; iStream++ ) {
        cuda(MemcpyAsync( 
                          hptrOut+iStream*streamStep, 
                          dptrOut+iStream*streamStep, 
                          streamStep*sizeof(float), 
                          cudaMemcpyDeviceToHost, 
                          streams[iStream] ) );
    }

    cuda(EventRecord( evStop, 0 ) );
    cuda(DeviceSynchronize() );
    chTimerGetTime( &chStop );
    *msWallClock = 1000.0f*chTimerElapsedTime( &chStart, &chStop );
    for ( size_t i = 0; i < N; i++ ) {
        if ( fabsf( hptrOut[i] - (alpha*hptrX[i]+hptrY[i]) ) > 1e-5f ) {
            status = cudaErrorUnknown;
            goto Error;
        }
    }
    cuda(EventElapsedTime( msTotal, evStart, evStop ) );
Error:
    if ( streams ) {
        for ( int i = 0; i < nStreams; i++ ) {
            cudaStreamDestroy( streams[i] );
        }
        delete[] streams;
    }
    cudaEventDestroy( evStart );
    cudaEventDestroy( evStop );
    cudaFree( dptrOut );
    cudaFree( dptrX );
    cudaFree( dptrY );
    cudaFreeHost( hptrOut );
    cudaFreeHost( hptrX );
    cudaFreeHost( hptrY );
    return status;
}

double
Bandwidth( float ms, double NumBytes )
{
    return NumBytes / (1000.0*ms);
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int N_Mfloats = 128;
    size_t N;
    int maxStreams = 8;
    int nBlocks = 1500;
    int nThreads = 256;
    float alpha = 2.0f;

    chCommandLineGet( &nBlocks, "nBlocks", argc, argv );
    chCommandLineGet( &nThreads, "nThreads", argc, argv );
    if ( ! chCommandLineGet( &N_Mfloats, "N", argc, argv ) ) {
        printf( "    Usage: use --N to specify number of Mfloats)\n");
    }
    printf( "Measuring times with %dM floats\n", N_Mfloats );
    if ( ! chCommandLineGet( &maxStreams, "maxStreams", argc, argv ) ) {
        printf( "Testing with default max of %d streams "
            "(set with --maxStreams <count>)\n", maxStreams );
    }
    printf( "\n" );

    N = 1048576*N_Mfloats;

    cuda(SetDeviceFlags( cudaDeviceMapHost ) );
    printf( "Streams\tTime (ms)\tMB/s\n" );
    for ( int numStreams = 1; numStreams <= maxStreams; numStreams++ ) {
        float msTotal, msWallClock;
        size_t thisN = (N / numStreams)*numStreams;
        CUDART_CHECK( MeasureTimes( &msTotal, &msWallClock, thisN, alpha, numStreams, nBlocks, nThreads ) );

        printf( "%d\t%.2f ms\t%.2f\n", numStreams, msTotal, Bandwidth( msWallClock, 3*thisN*sizeof(float) ) );
    }

Error:
    if ( status == cudaErrorMemoryAllocation ) {
        printf( "Memory allocation failed\n" );
    }
    else if ( cudaSuccess != status ) {
        printf( "Failed\n" );
    }
    return cudaSuccess != status;
}
