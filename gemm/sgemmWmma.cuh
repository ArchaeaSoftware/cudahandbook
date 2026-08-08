/*
 *
 * sgemmWmma.cuh
 *
 * WMMA inner loop shared by the two Tensor Core samples of The CUDA Handbook,
 * 2nd ed., Chapter 17: sgemm5Wmma (single-buffered) and sgemm6WmmaAsync
 * (cp.async double-buffered). Both march the same BK-deep slab of shared
 * memory through TF32 Tensor Core fragments; only the way the slab is staged
 * differs, so the compute lives here and the staging lives in each sample.
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

#ifndef __SGEMM_WMMA_CUH__
#define __SGEMM_WMMA_CUH__

#include <mma.h>
using namespace nvcuda;

//
// One BK-deep step of the Tensor Core inner loop (Listing 17-5).
//
// The names it reads must be in scope at the point of expansion:
//   BK, BN            -- slab depth and block-tile width
//   TMw, TNw          -- this warp's grid of 16x16 output tiles
//   warpRow, warpCol  -- the warp's tile origin within the block tile
//   cF[TMw][TNw]      -- FP32 accumulator fragments
// SRC_A is a [.][BK] row-major shared tile; SRC_B is [BK][.] row-major.
// The 16x16x8 tf32 shape consumes the slab eight columns of K at a time;
// inputs are rounded to TF32 before each mma_sync, and C accumulates in FP32.
//
#define WMMA_COMPUTE( SRC_A, SRC_B )                                                             \
    for ( int kk = 0; kk < BK; kk += 8 ) {                                                       \
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> aF[TMw];     \
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bF[TNw];     \
        _Pragma("unroll")                                                                        \
        for ( int i = 0; i < TMw; i++ ) {                                                        \
            wmma::load_matrix_sync( aF[i], &(SRC_A)[warpRow*(TMw*16)+i*16][kk], BK );             \
            _Pragma("unroll")                                                                    \
            for ( int t = 0; t < aF[i].num_elements; t++ )                                       \
                aF[i].x[t] = wmma::__float_to_tf32( aF[i].x[t] );                                 \
        }                                                                                        \
        _Pragma("unroll")                                                                        \
        for ( int j = 0; j < TNw; j++ ) {                                                        \
            wmma::load_matrix_sync( bF[j], &(SRC_B)[kk][warpCol*(TNw*16)+j*16], BN );             \
            _Pragma("unroll")                                                                    \
            for ( int t = 0; t < bF[j].num_elements; t++ )                                       \
                bF[j].x[t] = wmma::__float_to_tf32( bF[j].x[t] );                                 \
        }                                                                                        \
        _Pragma("unroll")                                                                        \
        for ( int i = 0; i < TMw; i++ ) {                                                        \
            _Pragma("unroll")                                                                    \
            for ( int j = 0; j < TNw; j++ )                                                      \
                wmma::mma_sync( cF[i][j], aF[i], bF[j], cF[i][j] );                               \
        }                                                                                        \
    }

#endif // __SGEMM_WMMA_CUH__
