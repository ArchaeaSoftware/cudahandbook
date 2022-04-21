/*
 *
 * bodybodyInteraction_AVX.h
 *
 * Intel x86/x86_64 AVX implementation of N-body computation.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
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

#include <immintrin.h>

inline void
bodyBodyInteraction_FMA(
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

    const __m256& softeningSquared,
    const __m256 mask )
{
    auto rcp_sqrt_nr_ps = [](const __m256 x) -> __m256 {
        const __m256
            nr      = _mm256_rsqrt_ps(x),
            muls    = _mm256_mul_ps(_mm256_mul_ps(nr, nr), x),
            beta    = _mm256_mul_ps(_mm256_set1_ps(0.5f), nr),
            gamma   = _mm256_sub_ps(_mm256_set1_ps(3.0f), muls);

        return _mm256_mul_ps(beta, gamma);
    };

    // r_01  [3 FLOPS]
    __m256 dx = _mm256_sub_ps( x1, x0 );
    __m256 dy = _mm256_sub_ps( y1, y0 );
    __m256 dz = _mm256_sub_ps( z1, z0 );

    // d^2 + e^2 [6 FLOPS]
    __m256 distSq = 
        _mm256_fmadd_ps( dx, dx, 
            _mm256_fmadd_ps( dy, dy,
                _mm256_fmadd_ps( dz, dz, softeningSquared ) ) );
#if 0
    __m256 distSq = 
        _mm256_add_ps( 
            _mm256_add_ps( 
                _mm256_mul_ps( dx, dx ), 
                _mm256_mul_ps( dy, dy ) 
            ), 
            _mm256_mul_ps( dz, dz ) 
        );
    distSq = _mm256_add_ps( distSq, softeningSquared );
#endif

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    __m256 invDist = rcp_sqrt_nr_ps( distSq );
    __m256 invDistCube = 
        _mm256_mul_ps( 
            invDist, 
            _mm256_mul_ps( 
                invDist, invDist ) 
        );

    // s = m_j * invDistCube [1 FLOP]
    __m256 s = _mm256_andnot_ps( mask, _mm256_mul_ps( mass1, invDistCube ) );

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    fx = _mm256_fmadd_ps( dx, s, fx );
    fy = _mm256_fmadd_ps( dy, s, fy );
    fz = _mm256_fmadd_ps( dz, s, fz );
#if 0
    fx = _mm256_add_ps( fx, _mm256_mul_ps( dx, s ) );
    fy = _mm256_add_ps( fy, _mm256_mul_ps( dy, s ) );
    fz = _mm256_add_ps( fz, _mm256_mul_ps( dz, s ) );
#endif
}

