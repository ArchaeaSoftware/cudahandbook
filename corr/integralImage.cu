/*
 *
 * integralImage.cu
 *
 * Driver for the summed-area table builds in sat.cuh. Builds both moment SATs
 * (sum and sum-of-squares) three ways -- naive, CUB, and the transpose-free
 * decoupled-look-back DLB -- across a range of image sizes, cross-checks every
 * table exact against brute-force box sums, and times them. The point it makes:
 * the library build (CUB) beats naive, but its scan+transpose is memory-bound on
 * the transpose; the custom single-pass DLB, which never transposes and stages a
 * uint32 intermediate, is several times faster at scale.
 *
 * Build: nvcc -O3 -arch=sm_86 -I ../chLib integralImage.cu -o integralImage
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
#include <cstdint>

#include <chError.h>
#include "sat.cuh"

// Host cross-check: satBox vs brute-force box sums (sq selects the squared moment).
static long
boxCheck( const int64_t *hS, const uint8_t *hImg, int W, int H, bool sq )
{
    long bad = 0;
    for ( int t = 0; t < 4000; t++ ) {
        int x0 = rand() % W;
        int y0 = rand() % H;
        int x1 = x0 + 1 + rand() % (W - x0);
        int y1 = y0 + 1 + rand() % (H - y0);
        int64_t box = satBox( hS, W, x0, y0, x1, y1 );
        int64_t ref = 0;
        for ( int y = y0; y < y1; y++ ) {
            for ( int x = x0; x < x1; x++ ) {
                int64_t v = hImg[(size_t)y*W + x];
                ref += sq ? v*v : v;
            }
        }
        if ( box != ref ) bad++;
    }
    return bad;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status_cudart;
    int ret = 1;
    const int sizes[] = { 512, 1024, 2048, 4096 };
    const int nSizes = (int)(sizeof(sizes)/sizeof(sizes[0]));
    const int Wmax = sizes[nSizes-1];
    const size_t Nmax = (size_t)Wmax*Wmax;
    const int iters = 50;

    uint8_t *hImg = NULL;
    uint8_t *dImg = NULL;
    int64_t *dSum = NULL;
    int64_t *dSumSq = NULL;
    int64_t *hS = NULL;
    void *scratch = NULL;
    cudaEvent_t e0 = NULL;
    cudaEvent_t e1 = NULL;

    size_t scratchBytes = satScratchBytes<SAT_CUB>( Wmax, Wmax );   // CUB needs the most
    if ( satScratchBytes<SAT_DLB>( Wmax, Wmax ) > scratchBytes ) {
        scratchBytes = satScratchBytes<SAT_DLB>( Wmax, Wmax );
    }

    srand( 5 );
    hImg = (uint8_t *) malloc( Nmax );
    hS   = (int64_t *) malloc( Nmax*sizeof(int64_t) );
    for ( size_t i = 0; i < Nmax; i++ ) {
        hImg[i] = (uint8_t)(rand() & 0xff);
    }

    cuda( Malloc( &dImg,    Nmax ) );
    cuda( Malloc( &dSum,    Nmax*sizeof(int64_t) ) );
    cuda( Malloc( &dSumSq,  Nmax*sizeof(int64_t) ) );
    cuda( Malloc( &scratch, scratchBytes ) );
    cuda( Memcpy( dImg, hImg, Nmax, cudaMemcpyHostToDevice ) );
    cuda( EventCreate( &e0 ) );
    cuda( EventCreate( &e1 ) );

    {
        cudaDeviceProp prop;
        cuda( GetDeviceProperties( &prop, 0 ) );
        printf( "%s  summed-area table build, both moments (naive / CUB / DLB), ms\n\n", prop.name );
        printf( "  %5s  %9s %9s %9s   %7s %7s  %6s\n",
                "size", "naive", "CUB", "DLB", "N/DLB", "CUB/DLB", "exact" );
    }

    for ( int si = 0; si < nSizes; si++ ) {
        int W = sizes[si];
        int H = W;
        size_t N = (size_t)W*H;
        float msN = 0.f;
        float msC = 0.f;
        float msD = 0.f;
        long bad = 0;

        #define BENCH(TAG,MS) do {                                                            \
            CUDART_CHECK( satBuild<TAG>( dImg, W, H, dSum, dSumSq, scratch, scratchBytes ) );  \
            cuda( DeviceSynchronize() );                                                      \
            cuda( Memcpy( hS, dSum,   N*sizeof(int64_t), cudaMemcpyDeviceToHost ) );           \
            bad += boxCheck( hS, hImg, W, H, false );                                         \
            cuda( Memcpy( hS, dSumSq, N*sizeof(int64_t), cudaMemcpyDeviceToHost ) );           \
            bad += boxCheck( hS, hImg, W, H, true );                                          \
            cuda( EventRecord( e0, 0 ) );                                                     \
            for ( int i = 0; i < iters; i++ ) {                                              \
                CUDART_CHECK( satBuild<TAG>( dImg, W, H, dSum, dSumSq, scratch, scratchBytes ) ); \
            }                                                                                 \
            cuda( EventRecord( e1, 0 ) );                                                     \
            cuda( EventSynchronize( e1 ) );                                                   \
            cuda( EventElapsedTime( &MS, e0, e1 ) );                                          \
            MS /= iters;                                                                      \
        } while (0)

        BENCH( SAT_NAIVE, msN );
        BENCH( SAT_CUB,   msC );
        BENCH( SAT_DLB,   msD );
        #undef BENCH

        printf( "  %5d  %9.4f %9.4f %9.4f   %6.2fx %6.2fx  %s\n",
                W, msN, msC, msD, msN/msD, msC/msD, bad ? "MISMATCH" : "exact" );
    }
    ret = 0;

Error_cudart:
    if ( e0 ) cudaEventDestroy( e0 );
    if ( e1 ) cudaEventDestroy( e1 );
    cudaFree( dImg );
    cudaFree( dSum );
    cudaFree( dSumSq );
    cudaFree( scratch );
    free( hImg );
    free( hS );
    return ret;
}
