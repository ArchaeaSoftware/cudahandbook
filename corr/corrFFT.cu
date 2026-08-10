/*
 *
 * corrFFT.cu
 *
 * Single-template normalized cross-correlation by FFT: an exhaustive search of
 * one template over the whole image. The expensive term is the numerator
 *     SumIT(u,v) = sum_ab  I(u+a, v+b) * T(a,b),
 * which the correlation theorem computes as
 *     SumIT = IFFT( FFT(I) . conj(FFT(T_padded)) ),
 * in O(N log N) INDEPENDENT of the template size -- the property that makes FFT
 * the method of choice for large templates (a spatial search is O(N * K), K the
 * template area). cuFFT is single precision, so the FFT numerator is approximate
 * (fine for localization); the denominator SumI/SumISq comes from the exact
 * int64 summed-area table in sat.cuh, built by the fused single-pass
 * decoupled-look-back method (SAT_FUSED), and the template's SumT/SumTSq are constants.
 *
 * The template is extracted from the image itself, so it correlates perfectly
 * (coefficient 1) at its own location; the sample finds the correlation peak to
 * confirm it is detected there, and cross-checks the FFT numerator against a
 * direct spatial one.
 *
 * --padWidth/--padHeight grow the working image (coins placed top-left, the rest
 * zero-padded -- there is no data dependency in the timing), so the full-match
 * cost can be swept to realistic sizes where the FFT numerator flattens and the
 * summed-area-table build dominates.
 *
 * Build: nvcc -O3 -arch=sm_86 -I ../chLib corrFFT.cu ../chLib/pgm.cu -lcufft
 * Run:   ./corrFFT [image.pgm] [--padWidth W --padHeight H]
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
#include <string.h>
#include <math.h>
#include <cstdint>

#include <cufft.h>
#include <chError.h>
#include <chCommandLine.h>
#include <pgm.h>
#include "sat.cuh"

// chError.h has no cuFFT variant; this mirrors its cublas() macro.
#define cufft( fn ) do {                                                        \
        status_cufft = (cufft##fn);                                             \
        if ( CUFFT_SUCCESS != status_cufft ) {                                  \
            fprintf( stderr, "cuFFT failure (%s:%d): %s returned %d\n",         \
                     __FILE__, __LINE__, #fn, (int)status_cufft );              \
            goto Error_cufft;                                                   \
        }                                                                       \
    } while (0)

static const int T = 16;                 // template edge; window K = T*T
static const int Kg = T*T;

__global__ void
copyToPad( const uint8_t *img, int W, int H, int Wp, float *pad )
{
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < W*H; i += gridDim.x*blockDim.x ) {
        int x = i % W;
        int y = i / W;
        pad[y*Wp + x] = (float) img[y*W + x];
    }
}

// Place the T x T template (extracted at (tx,ty)) at the origin of a zeroed pad.
__global__ void
fillTemplate( const uint8_t *img, int W, int Wp, int tx, int ty, float *tpad )
{
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < Kg; i += gridDim.x*blockDim.x ) {
        int a = i / T;
        int b = i % T;
        tpad[a*Wp + b] = (float) img[(ty+a)*W + (tx+b)];
    }
}

// C = F . conj(Tt);  (a+bi)(c-di) = (ac+bd) + (bc-ad)i.
__global__ void
mulConj( const cufftComplex *F, const cufftComplex *Tt, cufftComplex *C, int n )
{
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x ) {
        float a = F[i].x;
        float b = F[i].y;
        float c = Tt[i].x;
        float d = Tt[i].y;
        C[i].x = a*c + b*d;
        C[i].y = b*c - a*d;
    }
}

// NCC coefficient per output pixel: FFT numerator + summed-area-table denominator.
__global__ void
combine( const float *corrRaw, const int64_t *dSum, const int64_t *dSumSq,
         int64_t SumT, int64_t SumTSq, int cPix, float invN,
         int W, int Wp, int outW, int M, float *Corr )
{
    for ( int p = blockIdx.x*blockDim.x + threadIdx.x; p < M; p += gridDim.x*blockDim.x ) {
        int u = p % outW;
        int v = p / outW;
        float   SumIT  = corrRaw[v*Wp + u] * invN;            // FFT numerator (fp32)
        int64_t SumI   = satBox( dSum,   W, u, v, u+T, v+T );
        int64_t SumISq = satBox( dSumSq, W, u, v, u+T, v+T );
        int64_t vI = (int64_t)cPix*SumISq - SumI*SumI;        // exact int64 variances
        int64_t vT = (int64_t)cPix*SumTSq - SumT*SumT;
        float num = (float)cPix*SumIT - (float)(SumI*SumT);
        Corr[p] = (vI > 0 && vT > 0) ? num * rsqrtf((float)vI) * rsqrtf((float)vT) : 0.f;
    }
}

// Direct spatial numerator -- the exact-integer cross-check for the FFT one.
__global__ void
sumIT_spatial( const uint8_t *img, int W, int outW, int tx, int ty, int M, int *out )
{
    for ( int p = blockIdx.x*blockDim.x + threadIdx.x; p < M; p += gridDim.x*blockDim.x ) {
        int u = p % outW;
        int v = p / outW;
        int acc = 0;
        for ( int a = 0; a < T; a++ ) {
            for ( int b = 0; b < T; b++ ) {
                acc += (int)img[(v+a)*W + (u+b)] * (int)img[(ty+a)*W + (tx+b)];
            }
        }
        out[p] = acc;
    }
}

// Find the single template's correlation peak: max-with-index reduction over M.
__global__ void
findPeak( const float *Corr, int M, int *bestIdx, float *bestScore )
{
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    float bv = -2.f;
    int   bi = 0;
    for ( int p = threadIdx.x; p < M; p += blockDim.x ) {
        if ( Corr[p] > bv ) {
            bv = Corr[p];
            bi = p;
        }
    }
    sval[threadIdx.x] = bv;
    sidx[threadIdx.x] = bi;
    __syncthreads();
    for ( int stride = blockDim.x/2; stride > 0; stride >>= 1 ) {
        if ( threadIdx.x < stride && sval[threadIdx.x+stride] > sval[threadIdx.x] ) {
            sval[threadIdx.x] = sval[threadIdx.x+stride];
            sidx[threadIdx.x] = sidx[threadIdx.x+stride];
        }
        __syncthreads();
    }
    if ( 0 == threadIdx.x ) {
        bestScore[0] = sval[0];
        bestIdx[0]   = sidx[0];
    }
}

static int
nextPow2( int x )
{
    int p = 1;
    while ( p < x ) p <<= 1;
    return p;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status_cudart;
    cufftResult status_cufft;
    int ret = 1;
    const int iters = 50;
    const char *inputFilename = "coins.pgm";
    int padWidth = 0, padHeight = 0;
    if ( argc > 1 && '-' != argv[1][0] ) inputFilename = argv[1];
    chCommandLineGet( &padWidth,  "padWidth",  argc, argv );
    chCommandLineGet( &padHeight, "padHeight", argc, argv );

    uint8_t *hRaw = NULL, *dRaw = NULL, *hImg = NULL, *dImg = NULL;
    unsigned int sysPitch = 0, devPitch = 0;
    int W = 0, H = 0;
    float *dImgF = NULL, *dTmplF = NULL, *dCorrRaw = NULL, *dCorr = NULL;
    cufftComplex *dImgHat = NULL, *dTmplHat = NULL, *dCorrHat = NULL;
    int *dITspat = NULL, *dBestIdx = NULL;
    float *dBestScore = NULL;
    int64_t *dSum = NULL, *dSumSq = NULL;
    void *scratch = NULL;
    size_t scratchBytes = 0;
    float *hCorr = NULL, *hRawCorr = NULL;
    int *hITspat = NULL, hBestIdx = 0;
    float hBestScore = 0.f;
    cufftHandle planR2C = 0, planC2R = 0;
    cudaEvent_t e0 = NULL, e1 = NULL;
    float msFFT = 0.f, msSpat = 0.f, msSAT = 0.f, msFull = 0.f;
    double worstRel = 0.0, selfCoef = 0.0;

    if ( pgmLoad( inputFilename, &hRaw, &sysPitch, &dRaw, &devPitch, &W, &H ) ) {
        fprintf( stderr, "could not load %s\n", inputFilename );
        return 1;
    }
    const int W0 = W, H0 = H;                 // original (coins) dimensions
    if ( padWidth  > W ) W = padWidth;         // grow the working image; no data dependency
    if ( padHeight > H ) H = padHeight;
    const int outW = W - T + 1, outH = H - T + 1, M = outW*outH;
    const int Wp = nextPow2( W ), Hp = nextPow2( H );   // pad the FFT to a fast, wrap-free size
    const int Cw = Wp/2 + 1;
    const int tx = W0/3, ty = H0/3;            // planted template stays inside the coins region
    const float invN = 1.0f / ((float)Hp*Wp);
    const int itersSpat = ( W >= 1024 ) ? 3 : iters;    // the O(N*K) cross-check is slow at scale
    const int uSig = ( W0 - T > 0 ) ? W0 - T : outW;    // cross-check only where there is signal
    const int vSig = ( H0 - T > 0 ) ? H0 - T : outH;    // (the zero-padded remainder has SumIT==0)

    // Pack coins into the top-left of a dense, zero-padded WxH buffer; template sums.
    hImg = (uint8_t *) malloc( (size_t)W*H );
    memset( hImg, 0, (size_t)W*H );
    for ( int y = 0; y < H0; y++ ) {
        for ( int x = 0; x < W0; x++ ) {
            hImg[y*W + x] = hRaw[y*sysPitch + x];
        }
    }
    int64_t SumT = 0, SumTSq = 0;
    for ( int a = 0; a < T; a++ ) {
        for ( int b = 0; b < T; b++ ) {
            int v = hImg[(ty+a)*W + (tx+b)];
            SumT   += v;
            SumTSq += (int64_t)v*v;
        }
    }

    cuda( Malloc( &dImg,     (size_t)W*H ) );
    cuda( Malloc( &dImgF,    (size_t)Hp*Wp*sizeof(float) ) );
    cuda( Malloc( &dTmplF,   (size_t)Hp*Wp*sizeof(float) ) );
    cuda( Malloc( &dCorrRaw, (size_t)Hp*Wp*sizeof(float) ) );
    cuda( Malloc( &dCorr,    (size_t)M*sizeof(float) ) );
    cuda( Malloc( &dImgHat,  (size_t)Hp*Cw*sizeof(cufftComplex) ) );
    cuda( Malloc( &dTmplHat, (size_t)Hp*Cw*sizeof(cufftComplex) ) );
    cuda( Malloc( &dCorrHat, (size_t)Hp*Cw*sizeof(cufftComplex) ) );
    cuda( Malloc( &dSum,     (size_t)W*H*sizeof(int64_t) ) );
    cuda( Malloc( &dSumSq,   (size_t)W*H*sizeof(int64_t) ) );
    cuda( Malloc( &dITspat,  (size_t)M*sizeof(int) ) );
    cuda( Malloc( &dBestIdx,    sizeof(int) ) );
    cuda( Malloc( &dBestScore,  sizeof(float) ) );
    scratchBytes = satScratchBytes<SAT_FUSED>( W, H );
    cuda( Malloc( &scratch, scratchBytes ) );
    cuda( Memcpy( dImg, hImg, (size_t)W*H, cudaMemcpyHostToDevice ) );

    cuda( Memset( dImgF, 0, (size_t)Hp*Wp*sizeof(float) ) );
    copyToPad<<<256,256>>>( dImg, W, H, Wp, dImgF );
    cuda( Memset( dTmplF, 0, (size_t)Hp*Wp*sizeof(float) ) );
    fillTemplate<<<64,256>>>( dImg, W, Wp, tx, ty, dTmplF );

    cufft( Plan2d( &planR2C, Hp, Wp, CUFFT_R2C ) );
    cufft( Plan2d( &planC2R, Hp, Wp, CUFFT_C2R ) );

    // FFT numerator: transform image and template, multiply-conjugate, invert.
    #define FFT_NUMERATOR() do {                                        \
        cufft( ExecR2C( planR2C, dImgF,  dImgHat ) );                   \
        cufft( ExecR2C( planR2C, dTmplF, dTmplHat ) );                  \
        mulConj<<<256,256>>>( dImgHat, dTmplHat, dCorrHat, Hp*Cw );      \
        cufft( ExecC2R( planC2R, dCorrHat, dCorrRaw ) );                \
    } while (0)

    cuda( EventCreate( &e0 ) );
    cuda( EventCreate( &e1 ) );

    FFT_NUMERATOR();
    cuda( DeviceSynchronize() );
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < iters; i++ ) FFT_NUMERATOR();
    cuda( EventRecord( e1, 0 ) );
    cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msFFT, e0, e1 ) );  msFFT /= iters;

    #define SPATIAL() sumIT_spatial<<<(M+127)/128,128>>>( dImg, W, outW, tx, ty, M, dITspat )
    SPATIAL();
    cuda( DeviceSynchronize() );
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < itersSpat; i++ ) SPATIAL();
    cuda( EventRecord( e1, 0 ) );
    cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msSpat, e0, e1 ) );  msSpat /= itersSpat;

    // Denominator via the summed-area table (sat.cuh), then the coefficient and peak.
    CUDART_CHECK( satBuild<SAT_FUSED>( dImg, W, H, dSum, dSumSq, scratch, scratchBytes ) );
    combine<<<(M+255)/256,256>>>( dCorrRaw, dSum, dSumSq, SumT, SumTSq, Kg, invN, W, Wp, outW, M, dCorr );
    findPeak<<<1,256>>>( dCorr, M, dBestIdx, dBestScore );
    cuda( DeviceSynchronize() );

    // SAT denominator build (DLB), timed on its own.
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < iters; i++ ) {
        satBuild<SAT_FUSED>( dImg, W, H, dSum, dSumSq, scratch, scratchBytes );
    }
    cuda( EventRecord( e1, 0 ) );
    cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msSAT, e0, e1 ) );  msSAT /= iters;

    // Full one-template pipeline: FFT numerator + SAT build + combine.
    #define FULL() do {                                                                   \
        FFT_NUMERATOR();                                                                  \
        satBuild<SAT_FUSED>( dImg, W, H, dSum, dSumSq, scratch, scratchBytes );              \
        combine<<<(M+255)/256,256>>>( dCorrRaw, dSum, dSumSq, SumT, SumTSq, Kg, invN, W, Wp, outW, M, dCorr ); \
    } while (0)
    FULL();
    cuda( DeviceSynchronize() );
    cuda( EventRecord( e0, 0 ) );
    for ( int i = 0; i < iters; i++ ) FULL();
    cuda( EventRecord( e1, 0 ) );
    cuda( EventSynchronize( e1 ) );
    cuda( EventElapsedTime( &msFull, e0, e1 ) );  msFull /= iters;
    #undef FFT_NUMERATOR
    #undef SPATIAL
    #undef FULL

    // ---- validation ----
    hCorr    = (float *) malloc( (size_t)M*sizeof(float) );
    hRawCorr = (float *) malloc( (size_t)Hp*Wp*sizeof(float) );
    hITspat  = (int *)   malloc( (size_t)M*sizeof(int) );
    cuda( Memcpy( hCorr,    dCorr,    (size_t)M*sizeof(float),   cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hRawCorr, dCorrRaw, (size_t)Hp*Wp*sizeof(float), cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( hITspat,  dITspat,  (size_t)M*sizeof(int),     cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( &hBestIdx,   dBestIdx,   sizeof(int),   cudaMemcpyDeviceToHost ) );
    cuda( Memcpy( &hBestScore, dBestScore, sizeof(float), cudaMemcpyDeviceToHost ) );

    // (a) FFT numerator vs the exact spatial one (worst relative error), sampled
    // over the coins region (uSig x vSig) where there is real signal.
    srand( 7 );
    for ( int s = 0; s < 400; s++ ) {
        int u = rand() % uSig;
        int v = rand() % vSig;
        double fft = (double)hRawCorr[v*Wp + u] * (double)invN;
        double sp  = (double)hITspat[v*outW + u];
        double rel = fabs( fft - sp ) / (fabs(sp) + 1.0);
        if ( rel > worstRel ) worstRel = rel;
    }
    // (b) the template's own location scores coefficient ~1.
    selfCoef = hCorr[(size_t)ty*outW + tx];
    // (c) the peak is at the planted location.
    {
        int px = hBestIdx % outW;
        int py = hBestIdx / outW;
        cudaDeviceProp prop;
        cuda( GetDeviceProperties( &prop, 0 ) );
        printf( "%s  cuFFT single-template NCC: image %dx%d (FFT pad %dx%d), template %dx%d, M=%d\n\n",
                prop.name, W, H, Wp, Hp, T, T, M );
        printf( "  %-26s %9s\n", "stage", "ms" );
        printf( "  %-26s %9.3f   cuFFT R2C,R2C,mulConj,C2R (fp32)\n", "FFT numerator", msFFT );
        printf( "  %-26s %9.3f   summed-area table, fused single-pass (exact int64)\n", "SAT denominator build", msSAT );
        printf( "  %-26s %9.3f   FFT numerator + SAT build + combine\n", "full pipeline", msFull );
        printf( "  %-26s %9.3f   direct spatial numerator, exact int, O(N*K)\n", "(spatial cross-check)", msSpat );
        printf( "\n  FFT vs spatial numerator: %.1fx faster (K=%dx%d); worst rel err %.2e (fp32 FFT)\n",
                msSpat/msFFT, T, T, worstRel );
        printf( "  self-corr at planted location: %.5f; peak (%d,%d) vs planted (%d,%d) -> %s\n",
                selfCoef, px, py, tx, ty, (px == tx && py == ty) ? "MATCH" : "MISLOCATED" );
    }
    ret = 0;

Error_cufft:
Error_cudart:
    if ( planR2C ) cufftDestroy( planR2C );
    if ( planC2R ) cufftDestroy( planC2R );
    if ( e0 ) cudaEventDestroy( e0 );
    if ( e1 ) cudaEventDestroy( e1 );
    cudaFree( dImg );  cudaFree( dImgF );  cudaFree( dTmplF );  cudaFree( dCorrRaw );  cudaFree( dCorr );
    cudaFree( dImgHat );  cudaFree( dTmplHat );  cudaFree( dCorrHat );
    cudaFree( dSum );  cudaFree( dSumSq );  cudaFree( scratch );  cudaFree( dITspat );
    cudaFree( dBestIdx );  cudaFree( dBestScore );
    free( hImg );  free( hCorr );  free( hRawCorr );  free( hITspat );
    return ret;
}
