/*
 *
 * sgemm1Naive.cu
 *
 * Stage 1 of the SGEMM optimization journey in The CUDA Handbook, 2nd ed.,
 * Chapter 17 (Listing 17-1): the naive kernel. One thread computes one
 * element of C, reading a full row of A and column of B from global memory
 * with no reuse. It is correct and simple, and every later stage exists to
 * recover the arithmetic intensity this version throws away.
 *
 * Build with: nvcc -O3 -arch=sm_80 sgemm1Naive.cu -lcublas
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

#include "sgemm.h"

//
// One thread per C element. Grid-stride loops in both dimensions so any
// launch configuration covers a C of any size.
//
__global__ void
sgemm_naive( int M, int N, int K, const float *A, const float *B, float *C )
{
    for ( int row = blockIdx.y*blockDim.y + threadIdx.y; row < M; row += gridDim.y*blockDim.y ) {
        for ( int col = blockIdx.x*blockDim.x + threadIdx.x; col < N; col += gridDim.x*blockDim.x ) {
            float acc = 0.f;
            for ( int k = 0; k < K; k++ ) acc += A[row*K + k] * B[k*N + col];
            C[row*N + col] = acc;
        }
    }
}

int
main( int argc, char **argv )
{
    SgemmHarness h;
    h.init( argc, argv );

    dim3 block( 16, 16 );
    dim3 grid( (h.N + 15)/16, (h.M + 15)/16 );
    h.report( "naive", [&]{
        sgemm_naive<<<grid, block>>>( h.M, h.N, h.K, h.dA, h.dB, h.dC );
    } );

    h.reportCublas();
    h.teardown();
    return 0;
}
