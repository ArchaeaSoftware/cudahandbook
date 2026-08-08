/*
 *
 * sgemm5Wmma.cu
 *
 * Stage 5 of the SGEMM optimization journey in The CUDA Handbook, 2nd ed.,
 * Chapter 17 (Listing 17-5): Tensor Cores through the WMMA API. The block
 * stages BM x BK and BK x BN slabs into shared memory exactly as stage 4,
 * but the multiply-accumulate now runs on Tensor Cores: inputs are rounded
 * to TF32 and a single instruction evaluates a 16x16x8 tile. FP32 accumulate
 * keeps the result close to the strict-FP32 reference. The staging is still
 * synchronous; overlapping it is stage 6's job.
 *
 * Requires an Ampere-or-later GPU (sm_80+) for TF32 Tensor Cores.
 * Build with: nvcc -O3 -arch=sm_80 sgemm5Wmma.cu -lcublas
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
#include "sgemmWmma.cuh"

//
// BM x BN block tile, BK-deep, over a WM x WN grid of warps; each warp owns a
// (BM/WM) x (BN/WN) region, i.e. a TMw x TNw grid of 16x16 WMMA tiles. Slabs
// are staged with 16-byte (float4) copies. Launch WM*WN*32 threads and a
// (N/BN, M/BM) grid; requires M%BM == N%BN == K%BK == 0.
//
template<int BM, int BN, int BK, int WM, int WN>
__global__ void
sgemm_wmma( int M, int N, int K, const float *A, const float *B, float *C )
{
    const int nThreads = WM*WN*32, TMw = BM/(WM*16), TNw = BN/(WN*16);
    __shared__ __align__(16) float As[BM][BK];
    __shared__ __align__(16) float Bs[BK][BN];
    const int tid = threadIdx.x, warp = tid>>5, warpRow = warp/WN, warpCol = warp%WN;

    for ( int bRow = blockIdx.y*BM; bRow < M; bRow += gridDim.y*BM ) {
        for ( int bCol = blockIdx.x*BN; bCol < N; bCol += gridDim.x*BN ) {
            wmma::fragment<wmma::accumulator,16,16,8,float> cF[TMw][TNw];
            #pragma unroll
            for ( int i = 0; i < TMw; i++ ) {
                #pragma unroll
                for ( int j = 0; j < TNw; j++ ) wmma::fill_fragment( cF[i][j], 0.f );
            }
            for ( int k0 = 0; k0 < K; k0 += BK ) {
                #pragma unroll
                for ( int f = tid; f < BM*BK/4; f += nThreads ) {
                    int m = (f*4)/BK, k = (f*4)%BK;
                    *reinterpret_cast<float4*>( &As[m][k] ) =
                        *reinterpret_cast<const float4*>( &A[(bRow+m)*K + (k0+k)] );
                }
                #pragma unroll
                for ( int f = tid; f < BK*BN/4; f += nThreads ) {
                    int k = (f*4)/BN, n = (f*4)%BN;
                    *reinterpret_cast<float4*>( &Bs[k][n] ) =
                        *reinterpret_cast<const float4*>( &B[(k0+k)*N + (bCol+n)] );
                }
                __syncthreads();
                WMMA_COMPUTE( As, Bs )
                __syncthreads();
            }
            #pragma unroll
            for ( int i = 0; i < TMw; i++ ) {
                #pragma unroll
                for ( int j = 0; j < TNw; j++ )
                    wmma::store_matrix_sync(
                        &C[(bRow + warpRow*(TMw*16) + i*16)*N + (bCol + warpCol*(TNw*16) + j*16)],
                        cF[i][j], N, wmma::mem_row_major );
            }
        }
    }
}

int
main( int argc, char **argv )
{
    SgemmHarness h;
    h.init( argc, argv );

    // 128x64 block tile, 32-deep slabs, 2x2 warps => 128 threads.
    const int BM = 128, BN = 64, BK = 32, WM = 2, WN = 2;
    dim3 block( WM*WN*32 );
    dim3 grid( (h.N + BN - 1)/BN, (h.M + BM - 1)/BM );
    h.report( "WMMA / TF32 Tensor Cores (128x64)", [&]{
        sgemm_wmma<BM,BN,BK,WM,WN><<<grid, block>>>( h.M, h.N, h.K, h.dA, h.dB, h.dC );
    } );

    h.reportCublas();
    h.teardown();
    return 0;
}
