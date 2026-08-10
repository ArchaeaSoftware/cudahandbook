/*
 *
 * corrBankTC.cu
 *
 * Multi-template normalized cross-correlation on Tensor Cores.
 *
 * The single-template corr samples show that one template is a poor Tensor
 * Core fit (the GEMM's N dimension is 1). This sample matches the image
 * against a BANK of N templates at once: the expensive term
 *     SumIT[pixel][t] = sum_over_window  I(window) * T_t(window)
 * lowers (im2col) to a GEMM  C[M x N] = A[M x K] * B[K x N], where
 *   M = output pixels, K = template area, N = number of templates.
 * With N >= 16 that fills the tile, so it runs as an INT8 Tensor Core GEMM
 * via cuBLAS -- and INT8 keeps SumIT exact. Pixels are unsigned [0,255] but
 * CUDA_R_8I is signed, so the GEMM runs on (I-128),(T-128) and we undo the
 * offset with the identity
 *     SumIT = GEMM(I-128,T-128) + 128*SumI + 128*SumT - 128*128*K,
 * which folds into the coefficient combine (it already has SumI and SumT).
 *
 * The template bank is extracted from coins.pgm itself, so each template
 * correlates perfectly (coefficient == 1) at its own location, and the sample
 * finds each template's correlation peak to confirm it is detected there.
 *
 * The per-pixel denominator sums (SumI, SumISq) come from an int64_t summed-area
 * table (integral image, sat.cuh): a separable row-then-column inclusive scan, box sums
 * via the 4-corner rule in O(1)/pixel. int64_t keeps them exact -- a 512x512 8-bit
 * squared-sum already overflows int32.
 *
 * Requires an INT8-Tensor-Core GPU (sm_75+). Reference cuBLAS, not custom.
 * Build with: nvcc -O3 -arch=sm_75 -I ../chLib corrBankTC.cu ../chLib/pgm.cu -lcublas
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstdint>

#include <cublas_v2.h>       // before chError.h so the cublas() macro is defined
#include <chError.h>
#include <pgm.h>

#include "sat.cuh"

static const int T  = 16;               // template edge; window K = T*T = 256
static const int NT = 32;               // templates in the bank
static const int Kg = T*T;

// im2col of the packed WxH image into A[M x Kg] (unsigned, for the baseline and
// the sums) and As[M x Kg] (signed, = pixel-128, for the INT8 Tensor Cores).
__global__ void
im2col( const uint8_t *img, int W, int outW, int M,
        uint8_t *A, int8_t *As )
{
    for ( size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x; idx < (size_t)M*Kg;
              idx += (size_t)gridDim.x*blockDim.x ) {
        int p = idx / Kg, w = idx % Kg;
        int px = p % outW, py = p / outW;
        int wx = w % T,    wy = w / T;
        uint8_t v = img[(py+wy)*W + (px+wx)];
        A[idx] = v;  As[idx] = (int8_t)((int)v - 128);
    }
}

// Non-Tensor-Core baseline: one thread per pixel, reuse the window across templates.
__global__ void
sumIT_baseline( const uint8_t *A, const uint8_t *B, int M, int *C )
{
    for ( int p = blockIdx.x*blockDim.x + threadIdx.x; p < M; p += gridDim.x*blockDim.x ) {
        int acc[NT];
        #pragma unroll
        for ( int t = 0; t < NT; t++ ) acc[t] = 0;
        const uint8_t *arow = A + (size_t)p*Kg;
        for ( int w = 0; w < Kg; w++ ) {
            int iv = arow[w];
            const uint8_t *brow = B + (size_t)w*NT;
            #pragma unroll
            for ( int t = 0; t < NT; t++ ) acc[t] += iv * brow[t];
        }
        int *crow = C + (size_t)p*NT;
        #pragma unroll
        for ( int t = 0; t < NT; t++ ) crow[t] = acc[t];
    }
}

__global__ void
sumI_sq( const uint8_t *A, int M, int *SumI, int *SumISq )
{
    for ( int p = blockIdx.x*blockDim.x + threadIdx.x; p < M; p += gridDim.x*blockDim.x ) {
        const uint8_t *arow = A + (size_t)p*Kg;
        int s = 0, sq = 0;
        for ( int w = 0; w < Kg; w++ ) { int v = arow[w]; s += v; sq += v*v; }
        SumI[p] = s; SumISq[p] = sq;
    }
}

// SumI, SumISq per output pixel via the four-corner rule (satBox, sat.cuh):
// O(1) per pixel, independent of the template size.
__global__ void
windowSums( const int64_t *dSum, const int64_t *dSumSq, int W, int outW, int M,
            int *SumI, int *SumISq )
{
    for ( int p = blockIdx.x*blockDim.x + threadIdx.x; p < M; p += gridDim.x*blockDim.x ) {
        int x0 = p % outW;
        int y0 = p / outW;
        SumI[p]   = (int) satBox( dSum,   W, x0, y0, x0+T, y0+T );
        SumISq[p] = (int) satBox( dSumSq, W, x0, y0, x0+T, y0+T );
    }
}

// Template matching: for each template, find the pixel of maximum NCC
// coefficient -- one block per template, max-with-index reduction over pixels.
__global__ void
findPeaks( const float *Corr, int M, int *bestIdx, float *bestScore )
{
    int t = blockIdx.x;
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    float bv = -2.f; int bi = 0;
    for ( int p = threadIdx.x; p < M; p += blockDim.x ) {
        float c = Corr[(size_t)p*NT + t];
        if ( c > bv ) { bv = c; bi = p; }
    }
    sval[threadIdx.x] = bv; sidx[threadIdx.x] = bi;
    __syncthreads();
    for ( int stride = blockDim.x/2; stride > 0; stride >>= 1 ) {
        if ( threadIdx.x < stride && sval[threadIdx.x+stride] > sval[threadIdx.x] ) {
            sval[threadIdx.x] = sval[threadIdx.x+stride];
            sidx[threadIdx.x] = sidx[threadIdx.x+stride];
        }
        __syncthreads();
    }
    if ( threadIdx.x == 0 ) { bestScore[t] = sval[0]; bestIdx[t] = sidx[0]; }
}

// Reconstruct SumIT from the signed GEMM and produce the NCC coefficient.
__global__ void
combine( const int *rawIT, const int *SumI, const int *SumISq,
         const int *SumT, const int *SumTSq, int cPix, int M, float *Corr )
{
    for ( size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x; idx < (size_t)M*NT;
              idx += (size_t)gridDim.x*blockDim.x ) {
        int p = idx / NT, t = idx % NT;
        int64_t SumIT = (int64_t)rawIT[idx] + 128*SumI[p] + 128*SumT[t] - 128*128*cPix;
        // Numerator and the two variances are exact in int64 -- and int64 is
        // integer math, NOT the fp64 path that GeForce parts throttle. Doing
        // the subtractions in int64 also avoids the float cancellation a
        // straight-fp32 combine would suffer on low-variance windows. Only the
        // final ratio is fp32 (rsqrtf of each variance -- also sidesteps the
        // vI*vT product, which overflows for large templates).
        int64_t num = (int64_t)cPix*SumIT      - (int64_t)SumI[p]*SumT[t];
        int64_t vI  = (int64_t)cPix*SumISq[p]  - (int64_t)SumI[p]*SumI[p];
        int64_t vT  = (int64_t)cPix*SumTSq[t]  - (int64_t)SumT[t]*SumT[t];
        Corr[idx] = (vI > 0 && vT > 0)
                  ? (float)num * rsqrtf((float)vI) * rsqrtf((float)vT) : 0.f;
    }
}

int
main( int argc, char *argv[] )
{
    cudaError_t    status_cudart;
    cublasStatus_t status_cublas;
    int ret = 1;
    int alpha = 1, beta = 0;
    const int iters = 50;
    float msTC = 0.f, msB = 0.f;
    int64_t mism = 0, maxd = 0;
    double worstSelf = 1.0;
    int64_t satBad = 0;
    int mislocated = 0;
    double worstPeak = 1.0;
    int *hITb = NULL;

    const char *inputFilename = (argc > 1) ? argv[1] : "coins.pgm";
    uint8_t *hRaw = NULL, *dRaw = NULL, *hImg = NULL;
    unsigned int   sysPitch = 0, devPitch = 0;
    int W = 0, H = 0;

    uint8_t *dImg = NULL, *dA = NULL, *dBank = NULL;
    int8_t   *dAS = NULL,  *dBankS = NULL;
    int *dRawIT = NULL, *dITb = NULL, *dSumI = NULL, *dSumISq = NULL, *dSumT = NULL, *dSumTSq = NULL;
    float *dCorr = NULL;
    int64_t *dSum = NULL, *dSumSq = NULL;
    int *dSumIchk = NULL, *dSumISqchk = NULL;
    int *dBestIdx = NULL, *hBestIdx = NULL;
    float *dBestScore = NULL, *hBestScore = NULL;
    uint8_t *hBank = NULL; int8_t *hBankS = NULL;
    int *hSumT = NULL, *hSumTSq = NULL, *tpx = NULL, *tpy = NULL;
    float *hCorr = NULL; int *hRawITh = NULL, *hSumIh = NULL;
    int *hIsq_sat = NULL, *hI_win = NULL, *hIsq_win = NULL;
    cublasHandle_t cb = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL;

    if ( pgmLoad( inputFilename, &hRaw, &sysPitch, &dRaw, &devPitch, &W, &H ) ) {
        fprintf( stderr, "could not load %s\n", inputFilename );
        return 1;
    }
    const int outW = W - T + 1, outH = H - T + 1, M = outW*outH;

    // Pack the (possibly pitched) host image into a dense WxH buffer.
    hImg = (uint8_t *) malloc( (size_t)W*H );
    for ( int y = 0; y < H; y++ ) {
        for ( int x = 0; x < W; x++ ) hImg[y*W+x] = hRaw[y*sysPitch + x];
    }

    // Build the bank: NT templates on a grid of locations in the image.
    hBank   = (uint8_t *) malloc( (size_t)Kg*NT );
    hBankS  = (int8_t   *) malloc( (size_t)Kg*NT );
    hSumT   = (int *) malloc( NT*sizeof(int) );
    hSumTSq = (int *) malloc( NT*sizeof(int) );
    tpx     = (int *) malloc( NT*sizeof(int) );
    tpy     = (int *) malloc( NT*sizeof(int) );
    {
        int cols = 8, rows = (NT + cols - 1)/cols;
        for ( int t = 0; t < NT; t++ ) {
            int gc = t % cols, gr = t / cols;
            tpx[t] = 4 + gc * ((outW - 8) / (cols - 1));
            tpy[t] = 4 + gr * ((outH - 8) / (rows > 1 ? rows - 1 : 1));
            if ( tpx[t] > outW-1 ) tpx[t] = outW-1;
            if ( tpy[t] > outH-1 ) tpy[t] = outH-1;
            int st = 0, stsq = 0;
            for ( int wy = 0; wy < T; wy++ ) {
                for ( int wx = 0; wx < T; wx++ ) {
                    uint8_t v = hImg[(tpy[t]+wy)*W + (tpx[t]+wx)];
                    hBank[(wy*T+wx)*NT + t] = v;
                    hBankS[(wy*T+wx)*NT + t] = (int8_t)((int)v - 128);
                    st += v; stsq += v*v;
                }
            }
            hSumT[t] = st; hSumTSq[t] = stsq;
        }
    }

    cuda( Malloc( &dImg,   (size_t)W*H ) );
    cuda( Malloc( &dA,     (size_t)M*Kg ) );
    cuda( Malloc( &dAS,    (size_t)M*Kg ) );
    cuda( Malloc( &dBank,  (size_t)Kg*NT ) );
    cuda( Malloc( &dBankS, (size_t)Kg*NT ) );
    cuda( Malloc( &dRawIT, (size_t)M*NT*sizeof(int) ) );
    cuda( Malloc( &dITb,   (size_t)M*NT*sizeof(int) ) );
    cuda( Malloc( &dSumI,  (size_t)M*sizeof(int) ) );
    cuda( Malloc( &dSumISq,(size_t)M*sizeof(int) ) );
    cuda( Malloc( &dSumT,  NT*sizeof(int) ) );
    cuda( Malloc( &dSumTSq,NT*sizeof(int) ) );
    cuda( Malloc( &dCorr,  (size_t)M*NT*sizeof(float) ) );
    cuda( Malloc( &dSum,   (size_t)W*H*sizeof(int64_t) ) );
    cuda( Malloc( &dSumSq, (size_t)W*H*sizeof(int64_t) ) );
    cuda( Malloc( &dSumIchk,   (size_t)M*sizeof(int) ) );
    cuda( Malloc( &dSumISqchk, (size_t)M*sizeof(int) ) );
    cuda( Malloc( &dBestIdx,   NT*sizeof(int) ) );
    cuda( Malloc( &dBestScore, NT*sizeof(float) ) );
    cuda( Memcpy( dImg,   hImg,   (size_t)W*H,   cudaMemcpyHostToDevice ) );
    cuda( Memcpy( dBank,  hBank,  (size_t)Kg*NT, cudaMemcpyHostToDevice ) );
    cuda( Memcpy( dBankS, hBankS, (size_t)Kg*NT, cudaMemcpyHostToDevice ) );
    cuda( Memcpy( dSumT,  hSumT,  NT*sizeof(int),cudaMemcpyHostToDevice ) );
    cuda( Memcpy( dSumTSq,hSumTSq,NT*sizeof(int),cudaMemcpyHostToDevice ) );

    im2col<<<1024,256>>>( dImg, W, outW, M, dA, dAS );  cuda( GetLastError() );
    // Denominator sums via the integral image (sat.cuh, four-corner rule, O(1)/pixel).
    CUDART_CHECK( satBuild<SAT_NAIVE>( dImg, W, H, dSum, dSumSq, NULL, 0 ) );
    windowSums<<<(M+255)/256,256>>>( dSum, dSumSq, W, outW, M, dSumI, dSumISq );  cuda( GetLastError() );
    // Cross-check the integral-image sums against the direct window loop.
    sumI_sq<<<(M+255)/256,256>>>( dA, M, dSumIchk, dSumISqchk );  cuda( GetLastError() );
    cuda( DeviceSynchronize() );

    cublas( Create( &cb ) );
    cuda( EventCreate( &e0 ) ); cuda( EventCreate( &e1 ) );

    // Row-major C(MxN) = As(MxKg) * BankS(KgxN): hand cuBLAS (Bank, As) swapped.
    // Macros (not lambdas) so the cuda()/cublas() gotos never bypass an init.
    #define GEMM() cublasGemmEx( cb, CUBLAS_OP_N, CUBLAS_OP_N, NT, M, Kg, \
        &alpha, dBankS, CUDA_R_8I, NT, dAS, CUDA_R_8I, Kg, \
        &beta, dRawIT, CUDA_R_32I, NT, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP )
    #define BASE() sumIT_baseline<<<(M+127)/128,128>>>( dA, dBank, M, dITb )
    status_cublas = GEMM();
    if ( CUBLAS_STATUS_SUCCESS != status_cublas ) goto Error_cublas;
    cuda( DeviceSynchronize() );
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < iters; i++ ) GEMM();
    cuda( EventRecord( e1, 0 ) ); cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msTC, e0, e1 ) ); msTC /= iters;

    BASE(); cuda( DeviceSynchronize() );
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < iters; i++ ) BASE();
    cuda( EventRecord( e1, 0 ) ); cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msB, e0, e1 ) ); msB /= iters;
    #undef GEMM
    #undef BASE

    combine<<<(M*NT+255)/256,256>>>( dRawIT, dSumI, dSumISq, dSumT, dSumTSq, Kg, M, dCorr );
    cuda( GetLastError() ); cuda( DeviceSynchronize() );
    findPeaks<<<NT,256>>>( dCorr, M, dBestIdx, dBestScore );
    cuda( GetLastError() ); cuda( DeviceSynchronize() );

    // ---- validation ----
    hCorr    = (float *) malloc( (size_t)M*NT*sizeof(float) );
    hRawITh  = (int *) malloc( (size_t)M*NT*sizeof(int) );
    hSumIh   = (int *) malloc( (size_t)M*sizeof(int) );
    hITb = (int *) malloc( (size_t)M*NT*sizeof(int) );
    cuda( Memcpy( hCorr,   dCorr,  (size_t)M*NT*sizeof(float), cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hRawITh, dRawIT, (size_t)M*NT*sizeof(int),   cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hSumIh,  dSumI,  (size_t)M*sizeof(int),      cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hITb,    dITb,   (size_t)M*NT*sizeof(int),   cudaMemcpyDeviceToHost ) );

    // (a) SumIT (reconstructed) exactly matches the non-TC baseline.
    for ( size_t i = 0; i < (size_t)M*NT; i++ ) {
        int p = (int)(i/NT), t = (int)(i%NT);
        int64_t trueIT = (int64_t)hRawITh[i] + 128*hSumIh[p] + 128*hSumT[t] - 128*128*Kg;
        int64_t d = llabs( trueIT - hITb[i] ); if ( d ) { mism++; if ( d > maxd ) maxd = d; }
    }
    // (b) every template's self-location scores coefficient ~1.
    for ( int t = 0; t < NT; t++ ) {
        size_t p = (size_t)tpy[t]*outW + tpx[t];
        double c = hCorr[p*NT + t];
        if ( c < worstSelf ) worstSelf = c;
    }
    // (c) integral-image sums exactly match the direct window loop.
    hIsq_sat = (int *) malloc( (size_t)M*sizeof(int) );
    hI_win   = (int *) malloc( (size_t)M*sizeof(int) );
    hIsq_win = (int *) malloc( (size_t)M*sizeof(int) );
    cuda( Memcpy( hIsq_sat, dSumISq,    (size_t)M*sizeof(int), cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hI_win,   dSumIchk,   (size_t)M*sizeof(int), cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hIsq_win, dSumISqchk, (size_t)M*sizeof(int), cudaMemcpyDeviceToHost ) );
    for ( int p = 0; p < M; p++ )
        if ( hSumIh[p] != hI_win[p] || hIsq_sat[p] != hIsq_win[p] ) satBad++;
    // (d) template matching: each template's peak is at its planted location.
    hBestIdx   = (int *)   malloc( NT*sizeof(int) );
    hBestScore = (float *) malloc( NT*sizeof(float) );
    cuda( Memcpy( hBestIdx,   dBestIdx,   NT*sizeof(int),   cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hBestScore, dBestScore, NT*sizeof(float), cudaMemcpyDeviceToHost ) );
    for ( int t = 0; t < NT; t++ ) {
        int x = hBestIdx[t] % outW, y = hBestIdx[t] / outW;
        if ( x != tpx[t] || y != tpy[t] ) mislocated++;
        if ( hBestScore[t] < worstPeak ) worstPeak = hBestScore[t];
    }

    {
        cudaDeviceProp prop; cuda( GetDeviceProperties( &prop, 0 ) );
        double gop = 2.0*(double)M*NT*Kg/1e9;
        printf( "%s  sm_%d%d   image %dx%d, %d templates of %dx%d, M=%d pixels\n\n",
                prop.name, prop.major, prop.minor, W, H, NT, T, T, M );
        printf( "  %-26s %8s %9s\n", "SumIT method", "ms", "GOP/s" );
        printf( "  %-26s %8.3f %9.1f   INT8 Tensor Cores (cuBLAS)\n", "TC GEMM", msTC, gop/(msTC/1e3) );
        printf( "  %-26s %8.3f %9.1f   hand kernel, no TC\n",         "baseline", msB, gop/(msB/1e3) );
        printf( "\n  speedup: %.1fx\n", msB/msTC );
        printf( "  SumIT vs baseline: %lld mismatches (maxdiff %lld)\n", (long long)mism, (long long)maxd );
        printf( "  worst self-correlation coefficient: %.5f (want ~1.0)\n", worstSelf );
        printf( "  integral-image sums vs window loop: %lld mismatches\n", (long long)satBad );
        printf( "  template matching: %d/%d templates located at planted position (worst peak %.5f)\n",
                NT - mislocated, NT, worstPeak );
    }
    ret = 0;

Error_cublas:
Error_cudart:
    if ( cb ) cublasDestroy( cb );
    if ( e0 ) cudaEventDestroy( e0 );
    if ( e1 ) cudaEventDestroy( e1 );
    cudaFree( dImg ); cudaFree( dA ); cudaFree( dAS ); cudaFree( dBank ); cudaFree( dBankS );
    cudaFree( dRawIT ); cudaFree( dITb ); cudaFree( dSumI ); cudaFree( dSumISq );
    cudaFree( dSumT ); cudaFree( dSumTSq ); cudaFree( dCorr );
    cudaFree( dSum ); cudaFree( dSumSq ); cudaFree( dSumIchk ); cudaFree( dSumISqchk );
    cudaFree( dBestIdx ); cudaFree( dBestScore );
    return ret;
}
