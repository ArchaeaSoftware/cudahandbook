/*
 *
 * sgemm4RegBlock.cu
 *
 * Stage 4 of the SGEMM optimization journey in The CUDA Handbook, 2nd ed.,
 * Chapter 17 (Listing 17-4): shared-memory slabs combined with register
 * blocking. The block stages a BM x BK slab of A (stored transposed) and a
 * BK x BN slab of B into shared memory, and each thread computes a TM x TN
 * microtile of C from registers. This marries stage 3's cooperative staging
 * with stage 2's per-thread reuse and is the fastest of the FP32 kernels --
 * the payoff of raising reuse, not of shared memory by itself.
 *
 * Build with: nvcc -O3 -arch=sm_80 sgemm4RegBlock.cu -lcublas
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
// BM x BN block tile, BK-deep K-steps, each thread a TM x TN microtile.
// A's slab is stored transposed (As[k][m]) so the inner loop reads a column
// of A and a row of B with unit stride. Launch (BM/TM)*(BN/TN) threads and a
// (N/BN, M/BM) grid; requires M%BM == N%BN == K%BK == 0.
//
template<int BM, int BN, int BK, int TM, int TN>
__global__ void
sgemm_regblock( int M, int N, int K, const float *A, const float *B, float *C )
{
    const int nThreads = (BM/TM)*(BN/TN);
    __shared__ float As[BK][BM];   // transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN/TN), threadCol = tid % (BN/TN);
    const int rowA = tid / BK, colA = tid % BK, strideA = nThreads / BK;
    const int rowB = tid / BN, colB = tid % BN, strideB = nThreads / BN;

    for ( int bRow = blockIdx.y*BM; bRow < M; bRow += gridDim.y*BM ) {
        for ( int bCol = blockIdx.x*BN; bCol < N; bCol += gridDim.x*BN ) {
            float acc[TM][TN] = {};
            for ( int k0 = 0; k0 < K; k0 += BK ) {
                #pragma unroll
                for ( int off = 0; off < BM; off += strideA )
                    As[colA][rowA+off] = A[(bRow + rowA + off)*K + (k0 + colA)];
                #pragma unroll
                for ( int off = 0; off < BK; off += strideB )
                    Bs[rowB+off][colB] = B[(k0 + rowB + off)*N + (bCol + colB)];
                __syncthreads();
                #pragma unroll
                for ( int k = 0; k < BK; k++ ) {
                    float rA[TM], rB[TN];
                    #pragma unroll
                    for ( int i = 0; i < TM; i++ ) rA[i] = As[k][threadRow*TM + i];
                    #pragma unroll
                    for ( int j = 0; j < TN; j++ ) rB[j] = Bs[k][threadCol*TN + j];
                    #pragma unroll
                    for ( int i = 0; i < TM; i++ ) {
                        #pragma unroll
                        for ( int j = 0; j < TN; j++ ) acc[i][j] += rA[i]*rB[j];
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for ( int i = 0; i < TM; i++ ) {
                #pragma unroll
                for ( int j = 0; j < TN; j++ )
                    C[(bRow + threadRow*TM + i)*N + (bCol + threadCol*TN + j)] = acc[i][j];
            }
        }
    }
}

int
main( int argc, char **argv )
{
    SgemmHarness h;
    h.init( argc, argv );

    // 128x64 block tile, 8-deep K-steps, 8x8 microtiles => 128 threads.
    const int BM = 128, BN = 64, BK = 8, TM = 8, TN = 8;
    dim3 block( (BM/TM)*(BN/TN) );
    dim3 grid( (h.N + BN - 1)/BN, (h.M + BM - 1)/BM );
    h.report( "register-blocked (128x64)", [&]{
        sgemm_regblock<BM,BN,BK,TM,TN><<<grid, block>>>( h.M, h.N, h.K, h.dA, h.dB, h.dC );
    } );

    h.reportCublas();
    h.teardown();
    return 0;
}
