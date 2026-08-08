/*
 *
 * sgemm.h
 *
 * Shared host harness for the SGEMM optimization-journey samples of
 * The CUDA Handbook, 2nd ed., Chapter 17 (Matrix Multiplication).
 *
 * Each numbered sample (sgemm1Naive .. sgemm6WmmaAsync) defines one kernel
 * and drives these helpers to allocate the problem, fill A and B, time the
 * kernel over several launches, and check the result against a strict-FP32
 * cuBLAS reference. sgemmReportCublas() then prints cuBLAS FP32 and TF32 rows
 * so each stage shows its own gap to the vendor library.
 *
 * The matrices are row-major: C(MxN) = A(MxK) * B(KxN).
 *
 * Error handling follows the book's house convention: the cuda() and cublas()
 * paste-macros from <chError.h> goto Error_cudart / Error_cublas, and
 * CUDART_CHECK() propagates a helper's cudaError_t to the caller's cleanup
 * block. <cublas_v2.h> is included before <chError.h> so the cublas() macro
 * is defined.
 *
 * Copyright (c) 2025-2026, Archaea Software, LLC.
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

#ifndef __SGEMM_H__
#define __SGEMM_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <functional>

#include <cuda_runtime.h>
#include <cublas_v2.h>     // before chError.h so the cublas() macro is defined
#include <chError.h>

struct SgemmProblem {
    int M, N, K, iters;
    float *dA, *dB, *dC, *dRef;   // device: A(MxK), B(KxN), result and FP32 reference (MxN)
    float *hRef, *hOut;           // host staging for the error check
    cublasHandle_t cb;
};

//
// Row-major C(MxN) = A(MxK)*B(KxN) is column-major C^T = B^T A^T, so hand cuBLAS
// (B, A) with swapped extents and it writes row-major C for free.
//
static inline cublasStatus_t
sgemmCublas( const SgemmProblem *pb, float *out )
{
    const float one = 1.f, zero = 0.f;
    return cublasSgemm( pb->cb, CUBLAS_OP_N, CUBLAS_OP_N, pb->N, pb->M, pb->K,
                        &one, pb->dB, pb->N, pb->dA, pb->K, &zero, out, pb->N );
}

//
// Size the problem from argv (defaults 4096 2048 4096), allocate, fill A and B
// with small exact-in-float values, create the cuBLAS handle, and compute the
// strict-FP32 reference into dRef. Prints the table header on success.
//
static cudaError_t
sgemmSetup( SgemmProblem *pb, int argc, char **argv )
{
    cudaError_t    status_cudart;
    cublasStatus_t status_cublas;
    cudaDeviceProp prop;
    float *h = NULL;
    size_t eA, eB, eC, i;

    pb->cb = NULL;
    pb->dA = pb->dB = pb->dC = pb->dRef = NULL;
    pb->hRef = pb->hOut = NULL;
    pb->M = ( argc > 1 ) ? atoi( argv[1] ) : 4096;
    pb->N = ( argc > 2 ) ? atoi( argv[2] ) : 2048;
    pb->K = ( argc > 3 ) ? atoi( argv[3] ) : 4096;
    pb->iters = 50;

    eA = (size_t) pb->M*pb->K;
    eB = (size_t) pb->K*pb->N;
    eC = (size_t) pb->M*pb->N;
    cuda( Malloc( &pb->dA,   eA*sizeof(float) ) );
    cuda( Malloc( &pb->dB,   eB*sizeof(float) ) );
    cuda( Malloc( &pb->dC,   eC*sizeof(float) ) );
    cuda( Malloc( &pb->dRef, eC*sizeof(float) ) );
    pb->hRef = (float *) malloc( eC*sizeof(float) );
    pb->hOut = (float *) malloc( eC*sizeof(float) );
    h = (float *) malloc( (eA > eB ? eA : eB)*sizeof(float) );
    if ( ! pb->hRef || ! pb->hOut || ! h ) {
        status_cudart = cudaErrorMemoryAllocation;
        goto Error_cudart;
    }

    for ( i = 0; i < eA; i++ ) h[i] = (float) ((int)(i%13) - 6) * 0.125f;
    cuda( Memcpy( pb->dA, h, eA*sizeof(float), cudaMemcpyHostToDevice ) );
    for ( i = 0; i < eB; i++ ) h[i] = (float) ((int)(i%7) - 3) * 0.25f;
    cuda( Memcpy( pb->dB, h, eB*sizeof(float), cudaMemcpyHostToDevice ) );
    free( h ); h = NULL;

    cublas( Create( &pb->cb ) );
    cublas( SetMathMode( pb->cb, CUBLAS_PEDANTIC_MATH ) );   // strict FP32 reference
    status_cublas = sgemmCublas( pb, pb->dRef );
    if ( CUBLAS_STATUS_SUCCESS != status_cublas ) goto Error_cublas;
    cuda( DeviceSynchronize() );

    cuda( GetDeviceProperties( &prop, 0 ) );
    printf( "%s  sm_%d%d   C(%dx%d) = A(%dx%d) * B(%dx%d)   (%.1f GFLOP/call, iters=%d)\n\n",
            prop.name, prop.major, prop.minor, pb->M, pb->N, pb->M, pb->K, pb->K, pb->N,
            2.0*pb->M*pb->N*pb->K/1e9, pb->iters );
    printf( "  %-33s %10s %10s %9s\n", "stage", "ms", "GFLOP/s", "max err" );
    return cudaSuccess;

Error_cublas:
    fprintf( stderr, "cuBLAS setup failure (line %d): status %d\n",
             __LINE__, (int) status_cublas );
    free( h );
    return cudaErrorUnknown;
Error_cudart:
    free( h );
    return status_cudart;
}

//
// Time `launch` over pb->iters launches (after one warm-up), run one clean
// pass, and check pb->dC against the FP32 reference; print the
// "name  ms  GFLOP/s  max err" row.
//
static cudaError_t
sgemmReport( SgemmProblem *pb, const char *name, std::function<void()> launch )
{
    cudaError_t status_cudart;
    cudaEvent_t evStart = NULL, evStop = NULL;
    float ms = 0;
    double maxerr = 0;
    size_t eC = (size_t) pb->M*pb->N, i;

    cuda( EventCreate( &evStart ) );
    cuda( EventCreate( &evStop ) );

    launch();                                   // warm-up
    cuda( GetLastError() );                     // catch an invalid launch configuration
    cuda( DeviceSynchronize() );

    cuda( EventRecord( evStart, 0 ) );
    for ( i = 0; i < (size_t) pb->iters; i++ ) launch();
    cuda( EventRecord( evStop, 0 ) );
    cuda( EventSynchronize( evStop ) );
    cuda( EventElapsedTime( &ms, evStart, evStop ) );
    ms /= pb->iters;

    launch();                                   // one clean pass for the error check
    cuda( DeviceSynchronize() );
    cuda( Memcpy( pb->hRef, pb->dRef, eC*sizeof(float), cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( pb->hOut, pb->dC,   eC*sizeof(float), cudaMemcpyDeviceToHost ) );
    for ( i = 0; i < eC; i++ ) {
        double d = fabs( (double) pb->hRef[i] - pb->hOut[i] );
        if ( d > maxerr ) maxerr = d;
    }
    printf( "  %-33s %10.3f %10.1f %9.2e\n", name, ms,
            2.0*pb->M*pb->N*pb->K/(ms/1e3)/1e9, maxerr );

    cudaEventDestroy( evStart );
    cudaEventDestroy( evStop );
    return cudaSuccess;

Error_cudart:
    if ( evStart ) cudaEventDestroy( evStart );
    if ( evStop )  cudaEventDestroy( evStop );
    return status_cudart;
}

//
// cuBLAS reference rows: FP32 (default) and TF32 Tensor Core math.
//
static cudaError_t
sgemmReportCublas( SgemmProblem *pb )
{
    cudaError_t    status_cudart;
    cublasStatus_t status_cublas;

    cublas( SetMathMode( pb->cb, CUBLAS_DEFAULT_MATH ) );
    CUDART_CHECK( sgemmReport( pb, "cuBLAS Sgemm (FP32, default)",
                               [&]{ sgemmCublas( pb, pb->dC ); } ) );
    cublas( SetMathMode( pb->cb, CUBLAS_TF32_TENSOR_OP_MATH ) );
    CUDART_CHECK( sgemmReport( pb, "cuBLAS Sgemm (TF32)",
                               [&]{ sgemmCublas( pb, pb->dC ); } ) );
    return cudaSuccess;

Error_cublas:
    fprintf( stderr, "cuBLAS math-mode failure (line %d): status %d\n",
             __LINE__, (int) status_cublas );
    return cudaErrorUnknown;
Error_cudart:
    return status_cudart;
}

static void
sgemmTeardown( SgemmProblem *pb )
{
    if ( pb->cb ) cublasDestroy( pb->cb );
    cudaFree( pb->dA ); cudaFree( pb->dB ); cudaFree( pb->dC ); cudaFree( pb->dRef );
    free( pb->hRef ); free( pb->hOut );
}

#endif // __SGEMM_H__
