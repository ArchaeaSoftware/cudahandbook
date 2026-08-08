/*
 *
 * sgemm3Tiled.cu
 *
 * Stage 3 of the SGEMM optimization journey in The CUDA Handbook, 2nd ed.,
 * Chapter 17 (Listing 17-3): the textbook shared-memory tiling. The block
 * cooperatively stages a BM x BK slab of A and a BK x BN slab of B into
 * shared memory, then every thread reads its inputs from there. This is the
 * canonical "first shared-memory GEMM" -- yet with one output per thread it
 * trails the register-blocked cache kernel of stage 2, which is the point:
 * shared memory alone is not the lever; reuse per thread is (see stage 4).
 *
 * Build with: nvcc -O3 -arch=sm_80 -I ../chLib sgemm3Tiled.cu -lcublas
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
// Block computes a BM x BN tile of C, one output per thread, marching K in
// BK-deep slabs staged through shared memory. BM, BN, BK are independent.
// Launch with (BN x BM) threads and a (N/BN, M/BM) grid.
//
template<int BM, int BN, int BK>
__global__ void
sgemm_tiled( int M, int N, int K, const float *A, const float *B, float *C )
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*BN + tx, nThreads = BM*BN;

    for ( int brow = blockIdx.y*BM; brow < M; brow += gridDim.y*BM ) {
        for ( int bcol = blockIdx.x*BN; bcol < N; bcol += gridDim.x*BN ) {
            float acc = 0.f;
            for ( int k0 = 0; k0 < K; k0 += BK ) {
                for ( int e = tid; e < BM*BK; e += nThreads )
                    As[e/BK][e%BK] = A[(brow + e/BK)*K + (k0 + e%BK)];
                for ( int e = tid; e < BK*BN; e += nThreads )
                    Bs[e/BN][e%BN] = B[(k0 + e/BN)*N + (bcol + e%BN)];
                __syncthreads();
                #pragma unroll
                for ( int k = 0; k < BK; k++ ) acc += As[ty][k] * Bs[k][tx];
                __syncthreads();
            }
            C[(brow + ty)*N + (bcol + tx)] = acc;
        }
    }
}

int
main( int argc, char **argv )
{
    cudaError_t status_cudart;
    SgemmProblem pb;
    int ret = 1;
    const int BM = 32, BN = 32, BK = 32;

    CUDART_CHECK( sgemmSetup( &pb, argc, argv ) );
    {
        dim3 block( BN, BM );
        dim3 grid( (pb.N + BN - 1)/BN, (pb.M + BM - 1)/BM );
        CUDART_CHECK( sgemmReport( &pb, "shared tiled (32x32x32)", [&]{
            sgemm_tiled<BM,BN,BK><<<grid, block>>>( pb.M, pb.N, pb.K, pb.dA, pb.dB, pb.dC );
        } ) );
    }
    CUDART_CHECK( sgemmReportCublas( &pb ) );
    ret = 0;

Error_cudart:
    sgemmTeardown( &pb );
    return ret;
}
