/*
 *
 * sgemm2Cache.cu
 *
 * Stage 2 of the SGEMM optimization journey in The CUDA Handbook, 2nd ed.,
 * Chapter 17 (Listing 17-2): cache blocking without shared memory. Each
 * thread computes a TM x TN microtile of C, so each value it loads from A
 * and B feeds several multiply-adds held in registers. Reuse alone -- no
 * __shared__, no __syncthreads() -- lifts throughput several times over the
 * naive kernel, leaning on the L1/L2 caches to serve the neighboring threads.
 *
 * Build with: nvcc -O3 -arch=sm_80 sgemm2Cache.cu -lcublas
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
// Each thread owns a TM x TN microtile of C. For each step of K it loads TM
// values of A and TN values of B into registers, then issues TM*TN FMAs --
// so a single global load feeds many multiplies. TM and TN are independent
// tile dimensions.
//
template<int TM, int TN>
__global__ void
sgemm_cache( int M, int N, int K, const float *A, const float *B, float *C )
{
    for ( int row0 = (blockIdx.y*blockDim.y + threadIdx.y)*TM; row0 < M;
              row0 += gridDim.y*blockDim.y*TM ) {
        for ( int col0 = (blockIdx.x*blockDim.x + threadIdx.x)*TN; col0 < N;
                  col0 += gridDim.x*blockDim.x*TN ) {
            float acc[TM][TN] = {};
            for ( int k = 0; k < K; k++ ) {
                float a[TM], b[TN];
                for ( int i = 0; i < TM; i++ ) a[i] = A[(row0+i)*K + k];
                for ( int j = 0; j < TN; j++ ) b[j] = B[k*N + (col0+j)];
                for ( int i = 0; i < TM; i++ ) {
                    for ( int j = 0; j < TN; j++ ) acc[i][j] += a[i]*b[j];
                }
            }
            for ( int i = 0; i < TM; i++ ) {
                for ( int j = 0; j < TN; j++ ) C[(row0+i)*N + (col0+j)] = acc[i][j];
            }
        }
    }
}

int
main( int argc, char **argv )
{
    SgemmHarness h;
    h.init( argc, argv );

    // 16x16 threads each computing an 8x8 microtile => a 128x128 tile per block.
    const int TM = 8, TN = 8;
    dim3 block( 16, 16 );
    dim3 grid( (h.N + 16*TN - 1)/(16*TN), (h.M + 16*TM - 1)/(16*TM) );
    h.report( "cache-blocked (8x8)", [&]{
        sgemm_cache<TM,TN><<<grid, block>>>( h.M, h.N, h.K, h.dA, h.dB, h.dC );
    } );

    h.reportCublas();
    h.teardown();
    return 0;
}
