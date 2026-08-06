/*
 *
 * sgemm.cu
 *
 * SGEMM optimization journey from The CUDA Handbook, 2nd ed.,
 * Chapter 17 (Matrix Multiplication). Computes a rectangular product
 *     C (MxN) = A (MxK) * B (KxN)                (row-major)
 * and benchmarks a progression of kernels against cuBLAS on the same GPU:
 *     naive
 *     cache-blocked          (register microtiles, no shared memory)
 *     shared-memory tiled    (textbook one-output-per-thread)
 *     register-blocked
 *     register-blocked + cp.async double-buffer
 *     TF32 WMMA              (Tensor Cores)
 *     TF32 WMMA + cp.async double-buffer
 *
 * Build:  nvcc -O3 -arch=sm_86 sgemm.cu -lcublas -o sgemm
 * Run:    ./sgemm [M] [N] [K]        (defaults 4096 2048 4096)
 *
 * Requires SM 8.0+ (Ampere) for TF32 WMMA and cp.async.
 *
 * Copyright (c) 2026, Archaea Software, LLC.
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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cublas_v2.h>
using namespace nvcuda;

#define CK(x)  do{ cudaError_t e=(x); if(e){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// ---- Stage 1: naive -- one thread per C element, no reuse ------------------
__global__ void sgemm_naive( int M, int N, int K, const float *A, const float *B, float *C )
{
    for ( int row = blockIdx.y*blockDim.y + threadIdx.y; row < M; row += gridDim.y*blockDim.y ) {
        for ( int col = blockIdx.x*blockDim.x + threadIdx.x; col < N; col += gridDim.x*blockDim.x ) {
            float acc = 0.f;
            for ( int k = 0; k < K; k++ ) acc += A[row*K + k] * B[k*N + col];
            C[row*N + col] = acc;
        }
    }
}

// ---- Stage 1b: cache-blocked -- CPU-style loop tiling, no shared memory -----
//   Each thread computes a TM x TN microtile of C, reading A/B straight from
//   global memory and relying on the L1/L2 caches for the reuse the block's
//   threads share. TM, TN are independent tile parameters.
template<int TM, int TN>
__global__ void sgemm_cache( int M, int N, int K, const float *A, const float *B, float *C )
{
    for ( int row0 = (blockIdx.y*blockDim.y + threadIdx.y)*TM; row0 < M; row0 += gridDim.y*blockDim.y*TM ) {
        for ( int col0 = (blockIdx.x*blockDim.x + threadIdx.x)*TN; col0 < N; col0 += gridDim.x*blockDim.x*TN ) {
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

// ---- Stage 2: shared-memory tiled (textbook first cut) --------------------
//   Block computes a BM x BN tile of C, one output per thread, marching K in
//   BK steps; BM, BN, BK are independent. Launch with BN x BM threads.
template<int BM, int BN, int BK>
__global__ void sgemm_tiled( int M, int N, int K, const float *A, const float *B, float *C )
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty*BN + tx, nThreads = BM*BN;
    for ( int brow = blockIdx.y*BM; brow < M; brow += gridDim.y*BM ) {
        for ( int bcol = blockIdx.x*BN; bcol < N; bcol += gridDim.x*BN ) {
            float acc = 0.f;
            for ( int k0 = 0; k0 < K; k0 += BK ) {
                for ( int e = tid; e < BM*BK; e += nThreads ) As[e/BK][e%BK] = A[(brow+e/BK)*K + (k0+e%BK)];
                for ( int e = tid; e < BK*BN; e += nThreads ) Bs[e/BN][e%BN] = B[(k0+e/BN)*N + (bcol+e%BN)];
                __syncthreads();
                #pragma unroll
                for ( int k = 0; k < BK; k++ ) acc += As[ty][k] * Bs[k][tx];
                __syncthreads();
            }
            C[(brow+ty)*N + (bcol+tx)] = acc;
        }
    }
}

// ---- Stage 3: register-blocked -- each thread computes a TM x TN microtile --
//   BM x BN block tile (independent dims), BK-deep K-steps, A slab stored
//   transposed in shared. N % BN == M % BM == K % BK == 0; (BM/TM)*(BN/TN) threads.
template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_regblock( int M, int N, int K, const float *A, const float *B, float *C )
{
    const int nThreads = (BM/TM)*(BN/TN);
    __shared__ float As[BK][BM];   // transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN/TN), threadCol = tid % (BN/TN);
    const int rowA = tid / BK,  colA = tid % BK,  strideA = nThreads / BK;
    const int rowB = tid / BN,  colB = tid % BN,  strideB = nThreads / BN;

    for ( int bRow = blockIdx.y*BM; bRow < M; bRow += gridDim.y*BM ) {
    for ( int bCol = blockIdx.x*BN; bCol < N; bCol += gridDim.x*BN ) {
        float acc[TM][TN] = {};
        for ( int k0 = 0; k0 < K; k0 += BK ) {
            #pragma unroll
            for ( int off = 0; off < BM; off += strideA )
                As[colA][rowA+off] = A[(bRow+rowA+off)*K + (k0+colA)];
            #pragma unroll
            for ( int off = 0; off < BK; off += strideB )
                Bs[rowB+off][colB] = B[(k0+rowB+off)*N + (bCol+colB)];
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
                C[(bRow+threadRow*TM+i)*N + (bCol+threadCol*TN+j)] = acc[i][j];
        }
    }
    }
}

// ---- Stage 4: register-blocked + cp.async double-buffer (scalar) -----------
template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_regblock_async( int M, int N, int K, const float *A, const float *B, float *C )
{
    const int nThreads = (BM/TM)*(BN/TN);
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN/TN), threadCol = tid % (BN/TN);
    const int rowA = tid / BK,  colA = tid % BK,  strideA = nThreads / BK;
    const int rowB = tid / BN,  colB = tid % BN,  strideB = nThreads / BN;

    for ( int bRow = blockIdx.y*BM; bRow < M; bRow += gridDim.y*BM ) {
    for ( int bCol = blockIdx.x*BN; bCol < N; bCol += gridDim.x*BN ) {
        float acc[TM][TN] = {};
        auto stage = [&]( int buf, int k0 ) {
            #pragma unroll
            for ( int off = 0; off < BM; off += strideA )
                __pipeline_memcpy_async( &As[buf][colA][rowA+off],
                                         &A[(bRow+rowA+off)*K + (k0+colA)], sizeof(float) );
            #pragma unroll
            for ( int off = 0; off < BK; off += strideB )
                __pipeline_memcpy_async( &Bs[buf][rowB+off][colB],
                                         &B[(k0+rowB+off)*N + (bCol+colB)], sizeof(float) );
            __pipeline_commit();
        };
        int buf = 0;
        stage( buf, 0 );
        for ( int k0 = 0; k0 < K; k0 += BK ) {
            __pipeline_wait_prior( 0 );
            __syncthreads();
            if ( k0 + BK < K ) stage( buf ^ 1, k0 + BK );
            #pragma unroll
            for ( int k = 0; k < BK; k++ ) {
                float rA[TM], rB[TN];
                #pragma unroll
                for ( int i = 0; i < TM; i++ ) rA[i] = As[buf][k][threadRow*TM + i];
                #pragma unroll
                for ( int j = 0; j < TN; j++ ) rB[j] = Bs[buf][k][threadCol*TN + j];
                #pragma unroll
                for ( int i = 0; i < TM; i++ ) {
                    #pragma unroll
                    for ( int j = 0; j < TN; j++ ) acc[i][j] += rA[i]*rB[j];
                }
            }
            __syncthreads();
            buf ^= 1;
        }
        #pragma unroll
        for ( int i = 0; i < TM; i++ ) {
            #pragma unroll
            for ( int j = 0; j < TN; j++ )
                C[(bRow+threadRow*TM+i)*N + (bCol+threadCol*TN+j)] = acc[i][j];
        }
    }
    }
}

// ---- Stage 4v: register-blocked + FLOAT4-vectorized cp.async double-buffer -
template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_regblock_async_vec( int M, int N, int K, const float *A, const float *B, float *C )
{
    const int nThreads = (BM/TM)*(BN/TN);
    __shared__ __align__(16) float As[2][BM][BK];   // non-transposed: As[m][k]
    __shared__ __align__(16) float Bs[2][BK][BN];
    const int tid = threadIdx.x;
    const int threadRow = tid / (BN/TN), threadCol = tid % (BN/TN);

    for ( int bRow = blockIdx.y*BM; bRow < M; bRow += gridDim.y*BM ) {
    for ( int bCol = blockIdx.x*BN; bCol < N; bCol += gridDim.x*BN ) {
        float acc[TM][TN] = {};
        auto stage = [&]( int buf, int k0 ) {
            #pragma unroll
            for ( int f = tid; f < BM*BK/4; f += nThreads ) { int m=(f*4)/BK, k=(f*4)%BK;
                __pipeline_memcpy_async( &As[buf][m][k], &A[(bRow+m)*K + (k0+k)], sizeof(float4) ); }
            #pragma unroll
            for ( int f = tid; f < BK*BN/4; f += nThreads ) { int k=(f*4)/BN, n=(f*4)%BN;
                __pipeline_memcpy_async( &Bs[buf][k][n], &B[(k0+k)*N + (bCol+n)], sizeof(float4) ); }
            __pipeline_commit();
        };
        int buf = 0;
        stage( buf, 0 );
        for ( int k0 = 0; k0 < K; k0 += BK ) {
            __pipeline_wait_prior( 0 );
            __syncthreads();
            if ( k0 + BK < K ) stage( buf ^ 1, k0 + BK );
            #pragma unroll
            for ( int k = 0; k < BK; k++ ) {
                float rA[TM], rB[TN];
                #pragma unroll
                for ( int i = 0; i < TM; i++ ) rA[i] = As[buf][threadRow*TM + i][k];
                #pragma unroll
                for ( int j = 0; j < TN; j++ ) rB[j] = Bs[buf][k][threadCol*TN + j];
                #pragma unroll
                for ( int i = 0; i < TM; i++ ) {
                    #pragma unroll
                    for ( int j = 0; j < TN; j++ ) acc[i][j] += rA[i]*rB[j];
                }
            }
            __syncthreads();
            buf ^= 1;
        }
        #pragma unroll
        for ( int i = 0; i < TM; i++ ) {
            #pragma unroll
            for ( int j = 0; j < TN; j++ )
                C[(bRow+threadRow*TM+i)*N + (bCol+threadCol*TN+j)] = acc[i][j];
        }
    }
    }
}

// ---- Stage 5/6: tuned WMMA TF32 (single- and cp.async double-buffered) ------
//   BM x BN block tile, BK-deep, WM x WN warp grid; each warp owns a
//   (BM/WM)x(BN/WN) region of WMMA tiles. float->TF32 in, FP32 accumulate.
#define WMMA_COMPUTE( SRC_A, SRC_B )                                                        \
    for ( int kk = 0; kk < BK; kk += 8 ) {                                                  \
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> aF[TMw];\
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bF[TNw];\
        _Pragma("unroll")                                                                   \
        for ( int i = 0; i < TMw; i++ ) {                                                   \
            wmma::load_matrix_sync( aF[i], &(SRC_A)[warpRow*(TMw*16)+i*16][kk], BK );        \
            _Pragma("unroll")                                                               \
            for ( int t = 0; t < aF[i].num_elements; t++ ) aF[i].x[t] = wmma::__float_to_tf32(aF[i].x[t]); \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for ( int j = 0; j < TNw; j++ ) {                                                   \
            wmma::load_matrix_sync( bF[j], &(SRC_B)[kk][warpCol*(TNw*16)+j*16], BN );        \
            _Pragma("unroll")                                                               \
            for ( int t = 0; t < bF[j].num_elements; t++ ) bF[j].x[t] = wmma::__float_to_tf32(bF[j].x[t]); \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for ( int i = 0; i < TMw; i++ ) {                                                   \
            _Pragma("unroll")                                                               \
            for ( int j = 0; j < TNw; j++ ) wmma::mma_sync( cF[i][j], aF[i], bF[j], cF[i][j] ); \
        }                                                                                   \
    }

template<int BM,int BN,int BK,int WM,int WN>
__global__ void sgemm_wmma( int M, int N, int K, const float *A, const float *B, float *C )
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
            for ( int f = tid; f < BM*BK/4; f += nThreads ) { int m=(f*4)/BK, k=(f*4)%BK;
                *reinterpret_cast<float4*>(&As[m][k]) = *reinterpret_cast<const float4*>(&A[(bRow+m)*K+(k0+k)]); }
            #pragma unroll
            for ( int f = tid; f < BK*BN/4; f += nThreads ) { int k=(f*4)/BN, n=(f*4)%BN;
                *reinterpret_cast<float4*>(&Bs[k][n]) = *reinterpret_cast<const float4*>(&B[(k0+k)*N+(bCol+n)]); }
            __syncthreads();
            WMMA_COMPUTE( As, Bs )
            __syncthreads();
        }
        #pragma unroll
        for ( int i = 0; i < TMw; i++ ) {
            #pragma unroll
            for ( int j = 0; j < TNw; j++ )
                wmma::store_matrix_sync( &C[(bRow+warpRow*(TMw*16)+i*16)*N + (bCol+warpCol*(TNw*16)+j*16)],
                                         cF[i][j], N, wmma::mem_row_major );
        }
    }
    }
}

template<int BM,int BN,int BK,int WM,int WN>
__global__ void sgemm_wmma_db( int M, int N, int K, const float *A, const float *B, float *C )
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
        auto stage = [&]( int buf, int k0 ) {
            #pragma unroll
            for ( int f = tid; f < BM*BK/4; f += nThreads ) { int m=(f*4)/BK, k=(f*4)%BK;
                __pipeline_memcpy_async( &As[buf][m][k], &A[(bRow+m)*K+(k0+k)], sizeof(float4) ); }
            #pragma unroll
            for ( int f = tid; f < BK*BN/4; f += nThreads ) { int k=(f*4)/BN, n=(f*4)%BN;
                __pipeline_memcpy_async( &Bs[buf][k][n], &B[(k0+k)*N+(bCol+n)], sizeof(float4) ); }
            __pipeline_commit();
        };
        int buf = 0;
        stage( buf, 0 );
        for ( int k0 = 0; k0 < K; k0 += BK ) {
            __pipeline_wait_prior( 0 );
            __syncthreads();
            if ( k0 + BK < K ) stage( buf ^ 1, k0 + BK );
            WMMA_COMPUTE( As[buf], Bs[buf] )
            __syncthreads();
            buf ^= 1;
        }
        #pragma unroll
        for ( int i = 0; i < TMw; i++ ) {
            #pragma unroll
            for ( int j = 0; j < TNw; j++ )
                wmma::store_matrix_sync( &C[(bRow+warpRow*(TMw*16)+i*16)*N + (bCol+warpCol*(TNw*16)+j*16)],
                                         cF[i][j], N, wmma::mem_row_major );
        }
    }
    }
}

// ---- Padding experiment: WMMA with a PAD skew on the shared leading dim ----
//   Scalar shared loads in both PAD=0 and PAD>0 so the only difference is the
//   fragment-load stride -- isolates whether padding relieves load_matrix_sync
//   bank conflicts. (Distinct from the vectorized-load sgemm_wmma above.)
template<int BM,int BN,int BK,int WM,int WN,int PAD>
__global__ void sgemm_wmma_pad( int M, int N, int K, const float *A, const float *B, float *C )
{
    const int nThreads = WM*WN*32, TMw = BM/(WM*16), TNw = BN/(WN*16);
    __shared__ float As[BM][BK+PAD];
    __shared__ float Bs[BK][BN+PAD];
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
            for ( int e = tid; e < BM*BK; e += nThreads ) As[e/BK][e%BK] = A[(bRow+e/BK)*K + (k0+e%BK)];
            for ( int e = tid; e < BK*BN; e += nThreads ) Bs[e/BN][e%BN] = B[(k0+e/BN)*N + (bCol+e%BN)];
            __syncthreads();
            #pragma unroll
            for ( int kk = 0; kk < BK; kk += 8 ) {
                wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> aF[TMw];
                wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bF[TNw];
                #pragma unroll
                for ( int i = 0; i < TMw; i++ ) {
                    wmma::load_matrix_sync( aF[i], &As[warpRow*(TMw*16)+i*16][kk], BK+PAD );
                    #pragma unroll
                    for ( int t = 0; t < aF[i].num_elements; t++ ) aF[i].x[t] = wmma::__float_to_tf32(aF[i].x[t]);
                }
                #pragma unroll
                for ( int j = 0; j < TNw; j++ ) {
                    wmma::load_matrix_sync( bF[j], &Bs[kk][warpCol*(TNw*16)+j*16], BN+PAD );
                    #pragma unroll
                    for ( int t = 0; t < bF[j].num_elements; t++ ) bF[j].x[t] = wmma::__float_to_tf32(bF[j].x[t]);
                }
                #pragma unroll
                for ( int i = 0; i < TMw; i++ ) {
                    #pragma unroll
                    for ( int j = 0; j < TNw; j++ ) wmma::mma_sync( cF[i][j], aF[i], bF[j], cF[i][j] );
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for ( int i = 0; i < TMw; i++ ) {
            #pragma unroll
            for ( int j = 0; j < TNw; j++ )
                wmma::store_matrix_sync( &C[(bRow+warpRow*(TMw*16)+i*16)*N + (bCol+warpCol*(TNw*16)+j*16)],
                                         cF[i][j], N, wmma::mem_row_major );
        }
    }
    }
}

static float time_ms( std::function<void()> fn, int iters )
{
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    fn(); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for ( int i = 0; i < iters; i++ ) fn();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms/iters;
}

int main( int argc, char **argv )
{
    int M = ( argc > 1 ) ? atoi(argv[1]) : 4096;
    int N = ( argc > 2 ) ? atoi(argv[2]) : 2048;
    int K = ( argc > 3 ) ? atoi(argv[3]) : 4096;
    int iters = 50;

    size_t eA=(size_t)M*K, eB=(size_t)K*N, eC=(size_t)M*N;
    float *dA,*dB,*dC,*dRef;
    CK(cudaMalloc(&dA,eA*4)); CK(cudaMalloc(&dB,eB*4));
    CK(cudaMalloc(&dC,eC*4)); CK(cudaMalloc(&dRef,eC*4));
    { size_t big = (eA>eB?eA:eB);
      float *h=(float*)malloc(big*4);
      for(size_t i=0;i<eA;i++) h[i]=(float)((i%13)-6)*0.125f;
      CK(cudaMemcpy(dA,h,eA*4,cudaMemcpyHostToDevice));
      for(size_t i=0;i<eB;i++) h[i]=(float)((i%7)-3)*0.25f;
      CK(cudaMemcpy(dB,h,eB*4,cudaMemcpyHostToDevice)); free(h); }

    cublasHandle_t cb; cublasCreate(&cb);
    const float one=1.f, zero=0.f;
    // row-major C(MxN)=A(MxK)*B(KxN)  <=>  column-major C^T = B^T A^T
    auto cublasRun = [&](float *out){ cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, N,M,K,
                        &one, dB,N, dA,K, &zero, out,N); };
    cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH);      // dRef = strict FP32
    cublasRun(dRef); CK(cudaDeviceSynchronize());

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    double flop = 2.0*M*N*K;
    printf("%s sm_%d%d  SGEMM  M=%d N=%d K=%d  (%.1f GFLOP/call, iters=%d)\n\n",
           p.name,p.major,p.minor, M,N,K, flop/1e9, iters);
    printf("  %-30s %10s %10s %9s\n","stage","ms","GFLOP/s","max err");

    auto maxerr = [&](float *out)->double{
        static float *hr=0,*ho=0; if(!hr){hr=(float*)malloc(eC*4);ho=(float*)malloc(eC*4);}
        cudaMemcpy(hr,dRef,eC*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(ho,out,eC*4,cudaMemcpyDeviceToHost);
        double m=0; for(size_t i=0;i<eC;i++){ double d=fabs((double)hr[i]-ho[i]); if(d>m)m=d; }
        return m;
    };
    auto report=[&](const char*name,std::function<void()> fn){
        float ms=time_ms(fn,iters); fn(); CK(cudaDeviceSynchronize());
        printf("  %-30s %10.3f %10.1f %9.2e\n",name,ms,flop/(ms/1e3)/1e9,maxerr(dC));
    };

    dim3 bn(16,16), gn((N+15)/16,(M+15)/16);
    report("1. naive", [&]{ sgemm_naive<<<gn,bn>>>(M,N,K,dA,dB,dC); });

    if ( M%128==0 && N%128==0 && K%128==0 ) {
        dim3 bc(16,16);
        report("1b cache-blocked 1x1", [&]{ sgemm_cache<1,1><<<dim3(N/16, M/16), bc>>>(M,N,K,dA,dB,dC); });
        report("1b cache-blocked 2x2", [&]{ sgemm_cache<2,2><<<dim3(N/32, M/32), bc>>>(M,N,K,dA,dB,dC); });
        report("1b cache-blocked 4x4", [&]{ sgemm_cache<4,4><<<dim3(N/64, M/64), bc>>>(M,N,K,dA,dB,dC); });
        report("1b cache-blocked 8x8", [&]{ sgemm_cache<8,8><<<dim3(N/128,M/128),bc>>>(M,N,K,dA,dB,dC); });
        report("1b cache-blocked 4x8", [&]{ sgemm_cache<4,8><<<dim3(N/128,M/64), bc>>>(M,N,K,dA,dB,dC); });
        report("1b cache-blocked 8x4", [&]{ sgemm_cache<8,4><<<dim3(N/64, M/128),bc>>>(M,N,K,dA,dB,dC); });

        report("2. shared tiled 32x32x32", [&]{ sgemm_tiled<32,32,32><<<dim3(N/32,M/32),dim3(32,32)>>>(M,N,K,dA,dB,dC); });

        dim3 gr(N/64, M/128);
        report("3. register-blocked 128x64", [&]{ sgemm_regblock<128,64,8,8,8><<<gr,128>>>(M,N,K,dA,dB,dC); });
        report("4. reg-blocked + cp.async",  [&]{ sgemm_regblock_async<128,64,8,8,8><<<gr,128>>>(M,N,K,dA,dB,dC); });
        report("4v. reg-blocked + cp.async(f4)", [&]{ sgemm_regblock_async_vec<128,64,8,8,8><<<gr,128>>>(M,N,K,dA,dB,dC); });

        report("5b wmma 128x64 2x2",   [&]{ sgemm_wmma<128,64,32,2,2>  <<<dim3(N/64,M/128),128>>> (M,N,K,dA,dB,dC); });
        report("5c wmma 64x128 2x2",   [&]{ sgemm_wmma<64,128,32,2,2>  <<<dim3(N/128,M/64),128>>> (M,N,K,dA,dB,dC); });
        report("5d wmma 128x128 4x2",  [&]{ sgemm_wmma<128,128,16,4,2> <<<dim3(N/128,M/128),256>>>(M,N,K,dA,dB,dC); });
        report("pad wmma 128x64 PAD=0", [&]{ sgemm_wmma_pad<128,64,32,2,2,0><<<dim3(N/64,M/128),128>>>(M,N,K,dA,dB,dC); });
        report("pad wmma 128x64 PAD=8", [&]{ sgemm_wmma_pad<128,64,32,2,2,8><<<dim3(N/64,M/128),128>>>(M,N,K,dA,dB,dC); });
        report("6b wmma+async 128x64", [&]{ sgemm_wmma_db<128,64,16,2,2>  <<<dim3(N/64,M/128),128>>> (M,N,K,dA,dB,dC); });
        report("6c wmma+async 128x128",[&]{ sgemm_wmma_db<128,128,16,4,2> <<<dim3(N/128,M/128),256>>>(M,N,K,dA,dB,dC); });
    }

    auto cublasReport = [&](const char *name, cublasMath_t mode){
        cublasSetMathMode(cb, mode);
        float ms = time_ms([&]{ cublasRun(dC); }, iters);
        cublasRun(dC); CK(cudaDeviceSynchronize());
        printf("  %-30s %10.3f %10.1f %9.2e\n", name, ms, flop/(ms/1e3)/1e9, maxerr(dC));
    };
    cublasReport("cuBLAS default math",     CUBLAS_DEFAULT_MATH);
    cublasReport("cuBLAS pedantic (FP32)",  CUBLAS_PEDANTIC_MATH);
    cublasReport("cuBLAS TF32 tensor-op",   CUBLAS_TF32_TENSOR_OP_MATH);

    cublasDestroy(cb);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dRef);
    return 0;
}
