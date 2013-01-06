/*
 *
 * stream2Streams.cu
 *
 * Formulation of stream1Async.cu that uses streams to overlap data
 * transfers and kernel processing.
 *
 * Build with: nvcc -I ../chLib stream2Streams.cu
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

#include <stdio.h>
#include <stdlib.h>

#include "saxpyCPU.h"

//
// saxpy global function adds x[i]*alpha to each element y[i]
// and writes the result to out[i].
//
// Due to low arithmetic density, this kernel is extremely bandwidth-bound.
//

__global__ void
saxpy( float *out, const float *x, const float *y, size_t N, float alpha )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        out[i] = alpha*x[i]+y[i];
    }
}

cudaError_t
MeasureTimes( 
    float *msTotal,
    size_t N, 
    float alpha,
    int nStreams,
    int nBlocks, 
    int nThreads )
{
    cudaError_t status;
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
        CUDART_CHECK( cudaStreamCreate( &streams[i] ) );
    }
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
    CUDART_CHECK( cudaEventCreate( &evStop ) );
    for ( size_t i = 0; i < N; i++ ) {
        hptrX[i] = (float) rand() / RAND_MAX;
        hptrY[i] = (float) rand() / RAND_MAX;
    }
    CUDART_CHECK( cudaEventRecord( evStart, 0 ) );

    for ( int iStream = 0; iStream < nStreams; iStream++ ) {
        CUDART_CHECK( cudaMemcpyAsync( dptrX+iStream*streamStep, hptrX+iStream*streamStep, streamStep*sizeof(float), cudaMemcpyHostToDevice, streams[iStream] ) );
        CUDART_CHECK( cudaMemcpyAsync( dptrY+iStream*streamStep, hptrY+iStream*streamStep, streamStep*sizeof(float), cudaMemcpyHostToDevice, streams[iStream] ) );
    }

    for ( int iStream = 0; iStream < nStreams; iStream++ ) {
        saxpy<<<nBlocks, nThreads, 0, streams[iStream]>>>( dptrOut+iStream*streamStep, dptrX+iStream*streamStep, dptrY+iStream*streamStep, streamStep, alpha );
    }

    for ( int iStream = 0; iStream < nStreams; iStream++ ) {
        CUDART_CHECK( cudaMemcpyAsync( hptrOut+iStream*streamStep, dptrOut+iStream*streamStep, streamStep*sizeof(float), cudaMemcpyDeviceToHost, streams[iStream] ) );
    }

    CUDART_CHECK( cudaEventRecord( evStop, 0 ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    for ( size_t i = 0; i < N; i++ ) {
        if ( fabsf( hptrOut[i] - (alpha*hptrX[i]+hptrY[i]) ) > 1e-5f ) {
            status = cudaErrorUnknown;
            goto Error;
        }
    }
    CUDART_CHECK( cudaEventElapsedTime( msTotal, evStart, evStop ) );
Error:
    if ( streams ) {
        for ( int i = 0; i < nStreams; i++ ) {
            cudaStreamDestroy( streams[i] );
        }
        delete[] streams;
    }
    cudaEventDestroy( evStart );
    cudaEventDestroy( evStop );
    cudaFree( dptrX );
    cudaFree( dptrY );
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
    int nStreams = 8;
    int nBlocks = 1500;
    int nThreads = 256;
    float alpha = 2.0f;

    chCommandLineGet( &nBlocks, "nBlocks", argc, argv );
    chCommandLineGet( &nThreads, "nThreads", argc, argv );
    chCommandLineGet( &nThreads, "nStreams", argc, argv );
    chCommandLineGet( &N_Mfloats, "N", argc, argv );
    printf( "Measuring times with %dM floats", N_Mfloats );
    if ( N_Mfloats==128 ) {
        printf( " (use --N to specify number of Mfloats)");
    }
    printf( "\n" );

    N = 1048576*N_Mfloats;

    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    {
        float msTotal;
        CUDART_CHECK( MeasureTimes( &msTotal, N, alpha, nStreams, nBlocks, nThreads ) );
        printf( "Total time: %.2f ms (%.2f MB/s)\n", msTotal, Bandwidth( msTotal, 3*N*sizeof(float) ) );
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
