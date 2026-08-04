/*
 *
 * nbody_CPU_AVX_openmp.cpp
 *
 * Multithreaded AVX CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for AVX.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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

#ifdef __AVX__
#ifdef _OPENMP
#include <immintrin.h>

#include <chrono>

#include "nbody.h"
#include "bodybodyInteraction_AVX.h"
#include "nbody_CPU_SIMD.h"

float
ComputeGravitation_SIMD_openmp(
    float *force[3],
    float *pos[4],
    float *mass,
    float softeningSquared,
    size_t N
)
{
    // AVX processes eight bodies at a time; refuse a body count that is not
    // a multiple of eight rather than silently dropping the remainder.
    requireBodyCountForAVX( N );

    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();

    const __m256 softening = _mm256_set1_ps( softeningSquared );

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        __m256 ax = _mm256_setzero_ps();
        __m256 ay = _mm256_setzero_ps();
        __m256 az = _mm256_setzero_ps();
        __m256 x0 = _mm256_set1_ps( pos[0][i] );
        __m256 y0 = _mm256_set1_ps( pos[1][i] );
        __m256 z0 = _mm256_set1_ps( pos[2][i] );

        for ( int j = 0; j < N/8; j++ ) {

            bodyBodyInteraction(
                ax, ay, az,
                x0, y0, z0,
                _mm256_loadu_ps( pos[0] + 8*j ),
                _mm256_loadu_ps( pos[1] + 8*j ),
                _mm256_loadu_ps( pos[2] + 8*j ),
                _mm256_loadu_ps( mass   + 8*j ),
                softening );

        }
        // Sum the eight partial forces accumulated in each YMM register
        _mm_store_ss( &force[0][i], horizontal_sum_ps( ax ) );
        _mm_store_ss( &force[1][i], horizontal_sum_ps( ay ) );
        _mm_store_ss( &force[2][i], horizontal_sum_ps( az ) );
    }

    end = std::chrono::steady_clock::now();

    return (float) std::chrono::duration<double>(end - start).count() * 1000.0f;
}
#endif
#endif
