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
 * Build with: nvcc -O3 -arch=sm_80 -I ../chLib sgemm1Naive.cu -lcublas
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
    cudaError_t status_cudart;
    SgemmProblem pb;
    int ret = 1;

    CUDART_CHECK( sgemmSetup( &pb, argc, argv ) );
    {
        dim3 block( 16, 16 );
        dim3 grid( (pb.N + 15)/16, (pb.M + 15)/16 );
        CUDART_CHECK( sgemmReport( &pb, "naive", [&]{
            sgemm_naive<<<grid, block>>>( pb.M, pb.N, pb.K, pb.dA, pb.dB, pb.dC );
        } ) );
    }
    CUDART_CHECK( sgemmReportCublas( &pb ) );
    ret = 0;

Error_cudart:
    sgemmTeardown( &pb );
    return ret;
}
