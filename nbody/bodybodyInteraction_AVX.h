/*
 *
 * bodybodyInteraction_AVX.h
 *
 * Intel x86/x86_64 AVX implementation of N-body computation. AVX widens the
 * SSE registers to 256 bits, so eight bodies are processed per instruction
 * instead of four.
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
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
 *
 */

#include <immintrin.h>

//
// Full-precision reciprocal square root: refine the 12-bit VRSQRTPS estimate
// with one Newton-Raphson iteration, x1 = x0*(3 - a*x0^2)/2.
//
static inline __m256
rcp_sqrt_nr_ps(const __m256 x)
{
    const __m256
        nr    = _mm256_rsqrt_ps(x),
        muls  = _mm256_mul_ps(_mm256_mul_ps(nr, nr), x),
        beta  = _mm256_mul_ps(_mm256_set1_ps(0.5f), nr),
        gamma = _mm256_sub_ps(_mm256_set1_ps(3.0f), muls);

    return _mm256_mul_ps(beta, gamma);
}

//
// Sum the eight lanes of a YMM register: fold the high 128 bits onto the low
// 128 and reduce those four, leaving the total in the low lane.
//
static inline __m128
horizontal_sum_ps( const __m256 x )
{
    __m128 s = _mm_add_ps( _mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1) );
    s = _mm_add_ps( s, _mm_movehl_ps( s, s ) );
    return _mm_add_ss( s, _mm_shuffle_ps( s, s, 1 ) );
}

inline void
bodyBodyInteraction(
    __m256& fx,
    __m256& fy,
    __m256& fz,

    const __m256& x0,
    const __m256& y0,
    const __m256& z0,

    const __m256& x1,
    const __m256& y1,
    const __m256& z1,
    const __m256& mass1,

    const __m256& softeningSquared )
{
    // r_01  [3 FLOPS]
    __m256 dx = _mm256_sub_ps( x1, x0 );
    __m256 dy = _mm256_sub_ps( y1, y0 );
    __m256 dz = _mm256_sub_ps( z1, z0 );

    // d^2 + e^2 [6 FLOPS]
    __m256 distSq =
        _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps( dx, dx ),
                _mm256_mul_ps( dy, dy )
            ),
            _mm256_mul_ps( dz, dz )
        );
    distSq = _mm256_add_ps( distSq, softeningSquared );

    // invDistCube = 1/distSq^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    __m256 invDist = rcp_sqrt_nr_ps( distSq );
    __m256 invDistCube =
        _mm256_mul_ps(
            invDist,
            _mm256_mul_ps(
                invDist, invDist )
        );

    // s = m_j * invDistCube [1 FLOP]
    __m256 s = _mm256_mul_ps( mass1, invDistCube );

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    fx = _mm256_add_ps( fx, _mm256_mul_ps( dx, s ) );
    fy = _mm256_add_ps( fy, _mm256_mul_ps( dy, s ) );
    fz = _mm256_add_ps( fz, _mm256_mul_ps( dz, s ) );
}
