/*
 *
 * stream3Mapped.cu
 *
 * Formulation of stream1Async.cu that uses mapped pinned memory to
 * hold the input and output data.  Since the kernel can use mapped
 * pinned memory to initiate DMA transfers across the bus, this
 * version is simpler (no cudaMemcpy() calls) and just as fast.
 *
 * Build with: nvcc -I ../chLib stream3Mapped.cu
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
        out[i] += alpha*x[i]+y[i];
    }
}

void
saxpyCPU( float *out, const float *x, const float *y, size_t N, float alpha )
{
    for ( size_t i = 0; i < N; i++ ) {
        out[i] += alpha*x[i]+y[i];
    }
}

cudaError_t
MeasureTimes( 
    float *msTotal,
    size_t N, 
    float alpha,
    int nBlocks, 
    int nThreads )
{
    cudaError_t status;
    float *dptrOut = 0, *hptrOut = 0;
    float *dptrY = 0, *hptrY = 0;
    float *dptrX = 0, *hptrX = 0;
    cudaEvent_t evStart = 0;
    cudaEvent_t evStop = 0;

    CUDART_CHECK( cudaHostAlloc( &hptrOut, N*sizeof(float), cudaHostAllocMapped ) );
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrOut, hptrOut, 0 ) );
    memset( hptrOut, 0, N*sizeof(float) );
    CUDART_CHECK( cudaHostAlloc( &hptrY, N*sizeof(float), cudaHostAllocMapped ) );
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrY, hptrY, 0 ) );
    CUDART_CHECK( cudaHostAlloc( &hptrX, N*sizeof(float), cudaHostAllocMapped ) );
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrX, hptrX, 0 ) );

    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );
    for ( size_t i = 0; i < N; i++ ) {
        hptrX[i] = (float) rand() / RAND_MAX;
        hptrY[i] = (float) rand() / RAND_MAX;
    }
    CUDART_CHECK( cudaEventRecord( evStart, 0 ) );
        saxpy<<<nBlocks, nThreads>>>( dptrOut, dptrX, dptrY, N, alpha );
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
    cudaEventDestroy( evStop );
    cudaEventDestroy( evStart );
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
        float msTotal;
        CUDART_CHECK( MeasureTimes( &msTotal, N, alpha, nBlocks, nThreads ) );
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
