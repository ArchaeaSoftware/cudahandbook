/*
 *
 * sgemm.h
 *
 * Shared host harness for the SGEMM optimization-journey samples of
 * The CUDA Handbook, 2nd ed., Chapter 17 (Matrix Multiplication).
 *
 * Each numbered sample (sgemm1Naive .. sgemm6WmmaAsync) defines one kernel
 * and calls into this harness to allocate the problem, fill A and B, time
 * the kernel over several launches, and check the result against a strict-
 * FP32 cuBLAS reference. reportCublas() then prints cuBLAS FP32 and TF32
 * rows so each stage shows its own gap to the vendor library.
 *
 * The matrices are row-major: C(MxN) = A(MxK) * B(KxN).
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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_=(x); if (e_) { \
    printf( "CUDA error %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__ ); \
    exit(1); } } while (0)

//
// Test harness for one SGEMM stage. init() sizes the problem from argv
// (defaults 4096 2048 4096), fills A and B with small exact-in-float values,
// and computes a strict-FP32 cuBLAS reference into dRef. report() times a
// kernel launch and prints its throughput and worst-case error versus dRef.
//
struct SgemmHarness {
    int M, N, K, iters;
    float *dA, *dB, *dC, *dRef;   // device: A(MxK), B(KxN), C/result and reference (MxN)
    float *hRef, *hOut;           // host staging for the error check
    cublasHandle_t cb;

    // row-major C(MxN) = A(MxK)*B(KxN) is column-major C^T = B^T A^T, so hand
    // cuBLAS (B, A) with swapped extents and it writes row-major C for free.
    void cublasCompute( float *out ) {
        const float one = 1.f, zero = 0.f;
        cublasSgemm( cb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &one, dB, N, dA, K, &zero, out, N );
    }

    void init( int argc, char **argv ) {
        M     = ( argc > 1 ) ? atoi( argv[1] ) : 4096;
        N     = ( argc > 2 ) ? atoi( argv[2] ) : 2048;
        K     = ( argc > 3 ) ? atoi( argv[3] ) : 4096;
        iters = 50;

        size_t eA = (size_t) M*K, eB = (size_t) K*N, eC = (size_t) M*N;
        CK( cudaMalloc( &dA,   eA*sizeof(float) ) );
        CK( cudaMalloc( &dB,   eB*sizeof(float) ) );
        CK( cudaMalloc( &dC,   eC*sizeof(float) ) );
        CK( cudaMalloc( &dRef, eC*sizeof(float) ) );
        hRef = (float *) malloc( eC*sizeof(float) );
        hOut = (float *) malloc( eC*sizeof(float) );

        float *h = (float *) malloc( (eA > eB ? eA : eB)*sizeof(float) );
        for ( size_t i = 0; i < eA; i++ ) h[i] = (float) ((int)(i%13) - 6) * 0.125f;
        CK( cudaMemcpy( dA, h, eA*sizeof(float), cudaMemcpyHostToDevice ) );
        for ( size_t i = 0; i < eB; i++ ) h[i] = (float) ((int)(i%7) - 3) * 0.25f;
        CK( cudaMemcpy( dB, h, eB*sizeof(float), cudaMemcpyHostToDevice ) );
        free( h );

        cublasCreate( &cb );
        cublasSetMathMode( cb, CUBLAS_PEDANTIC_MATH );   // strict FP32 reference
        cublasCompute( dRef );
        CK( cudaDeviceSynchronize() );

        cudaDeviceProp p;
        cudaGetDeviceProperties( &p, 0 );
        printf( "%s  sm_%d%d   C(%dx%d) = A(%dx%d) * B(%dx%d)   (%.1f GFLOP/call, iters=%d)\n\n",
                p.name, p.major, p.minor, M, N, M, K, K, N, 2.0*M*N*K/1e9, iters );
        printf( "  %-33s %10s %10s %9s\n", "stage", "ms", "GFLOP/s", "max err" );
    }

    double gflops( float ms ) const { return 2.0*M*N*K / (ms/1e3) / 1e9; }

    // Mean milliseconds over `iters` launches, after one warm-up.
    float time( std::function<void()> fn ) {
        cudaEvent_t a, b;
        cudaEventCreate( &a ); cudaEventCreate( &b );
        fn(); CK( cudaDeviceSynchronize() );
        cudaEventRecord( a );
        for ( int i = 0; i < iters; i++ ) fn();
        cudaEventRecord( b ); cudaEventSynchronize( b );
        float ms = 0; cudaEventElapsedTime( &ms, a, b );
        cudaEventDestroy( a ); cudaEventDestroy( b );
        return ms / iters;
    }

    // Worst-case absolute difference of the last dC versus the FP32 reference.
    double maxerr() {
        size_t eC = (size_t) M*N;
        cudaMemcpy( hRef, dRef, eC*sizeof(float), cudaMemcpyDeviceToHost );
        cudaMemcpy( hOut, dC,   eC*sizeof(float), cudaMemcpyDeviceToHost );
        double m = 0;
        for ( size_t i = 0; i < eC; i++ ) {
            double d = fabs( (double) hRef[i] - hOut[i] );
            if ( d > m ) m = d;
        }
        return m;
    }

    void report( const char *name, std::function<void()> launch ) {
        float ms = time( launch );
        launch(); CK( cudaDeviceSynchronize() );      // one clean pass for the error check
        printf( "  %-33s %10.3f %10.1f %9.2e\n", name, ms, gflops(ms), maxerr() );
    }

    // cuBLAS reference rows: FP32 (default) and TF32 Tensor Core math.
    void reportCublas() {
        cublasSetMathMode( cb, CUBLAS_DEFAULT_MATH );
        report( "cuBLAS Sgemm (FP32, default)", [&]{ cublasCompute( dC ); } );
        cublasSetMathMode( cb, CUBLAS_TF32_TENSOR_OP_MATH );
        report( "cuBLAS Sgemm (TF32)",          [&]{ cublasCompute( dC ); } );
    }

    void teardown() {
        cublasDestroy( cb );
        cudaFree( dA ); cudaFree( dB ); cudaFree( dC ); cudaFree( dRef );
        free( hRef ); free( hOut );
    }
};

#endif // __SGEMM_H__
