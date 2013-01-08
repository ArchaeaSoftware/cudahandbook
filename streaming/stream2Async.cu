/*
 *
 * stream2Async.cu
 *
 * Microbenchmark to illustrate a bandwidth-limited workload.
 *
 * It separately measures the host->device transfer time, kernel
 * processing time, and device->host transfer time.  Due to low
 * arithmetic density in the saxpyGPU() kernel, the bulk of time
 * is spent transferring data. 
 *
 * Build with: nvcc -I ../chLib stream2Async.cu
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
    float *msHtoD, 
    float *msKernel, 
    float *msDtoH, 
    size_t N, 
    float alpha,
    int nBlocks, 
    int nThreads )
{
    cudaError_t status;
    chTimerTimestamp chStart, chStop;
    float *dptrOut = 0, *hptrOut = 0;
    float *dptrY = 0, *hptrY = 0;
    float *dptrX = 0, *hptrX = 0;
    cudaEvent_t evStart = 0;
    cudaEvent_t evHtoD = 0;
    cudaEvent_t evKernel = 0;
    cudaEvent_t evDtoH = 0;

    CUDART_CHECK( cudaHostAlloc( &hptrOut, N*sizeof(float), 0 ) );
    memset( hptrOut, 0, N*sizeof(float) );
    CUDART_CHECK( cudaHostAlloc( &hptrY, N*sizeof(float), 0 ) );
    CUDART_CHECK( cudaHostAlloc( &hptrX, N*sizeof(float), 0 ) );

    CUDART_CHECK( cudaMalloc( &dptrOut, N*sizeof(float) ) );
    CUDART_CHECK( cudaMemset( dptrOut, 0, N*sizeof(float) ) );

    CUDART_CHECK( cudaMalloc( &dptrY, N*sizeof(float) ) );
    CUDART_CHECK( cudaMemset( dptrY, 0, N*sizeof(float) ) );

    CUDART_CHECK( cudaMalloc( &dptrX, N*sizeof(float) ) );
    CUDART_CHECK( cudaMemset( dptrY, 0, N*sizeof(float) ) );

    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evHtoD ) );
    CUDART_CHECK( cudaEventCreate( &evKernel ) );
    CUDART_CHECK( cudaEventCreate( &evDtoH ) );
    for ( size_t i = 0; i < N; i++ ) {
        hptrX[i] = (float) rand() / RAND_MAX;
        hptrY[i] = (float) rand() / RAND_MAX;
    }

    //
    // begin timing
    //

    chTimerGetTime( &chStart );
    CUDART_CHECK( cudaEventRecord( evStart, 0 ) );
        CUDART_CHECK( cudaMemcpyAsync( dptrX, hptrX, N*sizeof(float), cudaMemcpyHostToDevice, NULL ) );
        CUDART_CHECK( cudaMemcpyAsync( dptrY, hptrY, N*sizeof(float), cudaMemcpyHostToDevice, NULL ) );
    CUDART_CHECK( cudaEventRecord( evHtoD, 0 ) );
        saxpyGPU<<<nBlocks, nThreads>>>( dptrOut, dptrX, dptrY, N, alpha );
    CUDART_CHECK( cudaEventRecord( evKernel, 0 ) );
        CUDART_CHECK( cudaMemcpyAsync( hptrOut, dptrOut, N*sizeof(float), cudaMemcpyDeviceToHost, NULL ) );
    CUDART_CHECK( cudaEventRecord( evDtoH, 0 ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    chTimerGetTime( &chStop );
    *msWallClock = 1000.0f*chTimerElapsedTime( &chStart, &chStop );

    //
    // end timing
    //

    for ( size_t i = 0; i < N; i++ ) {
        if ( fabsf( hptrOut[i] - (alpha*hptrX[i]+hptrY[i]) ) > 1e-5f ) {
            status = cudaErrorUnknown;
            goto Error;
        }
    }
    CUDART_CHECK( cudaEventElapsedTime( msHtoD, evStart, evHtoD ) );
    CUDART_CHECK( cudaEventElapsedTime( msKernel, evHtoD, evKernel ) );
    CUDART_CHECK( cudaEventElapsedTime( msDtoH, evKernel, evDtoH ) );
    CUDART_CHECK( cudaEventElapsedTime( msTotal, evStart, evDtoH ) );
Error:
    cudaEventDestroy( evDtoH );
    cudaEventDestroy( evKernel );
    cudaEventDestroy( evHtoD );
    cudaEventDestroy( evStart );
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
    int nBlocks = 1500;
    int nThreads = 256;
    float alpha = 2.0f;

    chCommandLineGet( &nBlocks, "nBlocks", argc, argv );
    chCommandLineGet( &nThreads, "nThreads", argc, argv );
    chCommandLineGet( &N_Mfloats, "N", argc, argv );
    printf( "Measuring times with %dM floats", N_Mfloats );
    if ( N_Mfloats==128 ) {
        printf( " (use --N to specify number of Mfloats)");
    }
    printf( "\n" );

    N = 1048576*N_Mfloats;

    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    {
        float msTotal, msWallClock, msHtoD, msKernel, msDtoH;
        CUDART_CHECK( MeasureTimes( &msTotal, &msWallClock, &msHtoD, &msKernel, &msDtoH, N, alpha, nBlocks, nThreads ) );
        printf( "Memcpy( host->device ): %.2f ms (%.2f MB/s)\n", msHtoD, Bandwidth( msHtoD, 2*N*sizeof(float) ) );
        printf( "Kernel processing     : %.2f ms (%.2f MB/s)\n", msKernel, Bandwidth( msKernel, 3*N*sizeof(float) ) );
        printf( "Memcpy (device->host ): %.2f ms (%.2f MB/s)\n\n", msDtoH, Bandwidth( msDtoH, N*sizeof(float) ) );
        printf( "Total time (wall clock): %.2f ms (%.2f MB/s)\n", msWallClock, Bandwidth( msWallClock, 3*N*sizeof(float) ) );
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
