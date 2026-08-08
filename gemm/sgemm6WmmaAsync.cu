/*
 *
 * sgemm6WmmaAsync.cu
 *
 * Stage 6 of the SGEMM optimization journey in The CUDA Handbook, 2nd ed.,
 * Chapter 17 (Listing 17-6): the WMMA kernel with cp.async double-buffering.
 * Tensor Cores retire a slab so quickly that the synchronous load of stage 5
 * is exposed on the critical path. Here two shared-memory buffers alternate:
 * while the Tensor Cores consume buf, cp.async streams the next slab into
 * buf^1 straight from global memory, and __pipeline_wait_prior() enforces the
 * interlock so the MMAs never read a slab before its copy has landed. This is
 * the case where asynchronous copy earns its keep -- there is dense compute to
 * hide the transfer behind, and the transfer was otherwise on the clock.
 *
 * Requires an Ampere-or-later GPU (sm_80+) for TF32 Tensor Cores and cp.async.
 * Build with: nvcc -O3 -arch=sm_80 sgemm6WmmaAsync.cu -lcublas
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
#include <cuda_pipeline.h>
#include "sgemmWmma.cuh"

//
// Same tiling as stage 5, but two shared-memory slabs alternate. Each slab is
// filled with 16-byte cp.async copies committed as one pipeline stage; the
// kernel waits on the prior stage before feeding the Tensor Cores, then issues
// the next stage's copies so they overlap the current slab's MMAs. Launch
// WM*WN*32 threads and a (N/BN, M/BM) grid; requires M%BM == N%BN == K%BK == 0.
//
template<int BM, int BN, int BK, int WM, int WN>
__global__ void
sgemm_wmma_db( int M, int N, int K, const float *A, const float *B, float *C )
{
    const int nThreads = WM*WN*32, TMw = BM/(WM*16), TNw = BN/(WN*16);
    __shared__ __align__(16) float As[2][BM][BK];
    __shared__ __align__(16) float Bs[2][BK][BN];
    const int tid = threadIdx.x, warp = tid>>5, warpRow = warp/WN, warpCol = warp%WN;

    for ( int bRow = blockIdx.y*BM; bRow < M; bRow += gridDim.y*BM ) {
        for ( int bCol = blockIdx.x*BN; bCol < N; bCol += gridDim.x*BN ) {
            wmma::fragment<wmma::accumulator,16,16,8,float> cF[TMw][TNw];
            #pragma unroll
            for ( int i = 0; i < TMw; i++ ) {
                #pragma unroll
                for ( int j = 0; j < TNw; j++ ) wmma::fill_fragment( cF[i][j], 0.f );
            }

            // Queue one slab's worth of async copies into buffer `buf`.
            auto stage = [&]( int buf, int k0 ) {
                #pragma unroll
                for ( int f = tid; f < BM*BK/4; f += nThreads ) {
                    int m = (f*4)/BK, k = (f*4)%BK;
                    __pipeline_memcpy_async( &As[buf][m][k], &A[(bRow+m)*K + (k0+k)], sizeof(float4) );
                }
                #pragma unroll
                for ( int f = tid; f < BK*BN/4; f += nThreads ) {
                    int k = (f*4)/BN, n = (f*4)%BN;
                    __pipeline_memcpy_async( &Bs[buf][k][n], &B[(k0+k)*N + (bCol+n)], sizeof(float4) );
                }
                __pipeline_commit();
            };

            int buf = 0;
            stage( buf, 0 );
            for ( int k0 = 0; k0 < K; k0 += BK ) {
                __pipeline_wait_prior( 0 );     // buf's copies have landed
                __syncthreads();
                if ( k0 + BK < K ) stage( buf ^ 1, k0 + BK );   // prefetch next slab
                WMMA_COMPUTE( As[buf], Bs[buf] )
                __syncthreads();
                buf ^= 1;
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

    // 128x64 block tile, 16-deep slabs (two buffered), 2x2 warps => 128 threads.
    const int BM = 128, BN = 64, BK = 16, WM = 2, WN = 2;
    dim3 block( WM*WN*32 );
    dim3 grid( (h.N + BN - 1)/BN, (h.M + BM - 1)/BM );
    h.report( "WMMA + cp.async double-buffer", [&]{
        sgemm_wmma_db<BM,BN,BK,WM,WN><<<grid, block>>>( h.M, h.N, h.K, h.dA, h.dB, h.dC );
    } );

    h.reportCublas();
    h.teardown();
    return 0;
}
