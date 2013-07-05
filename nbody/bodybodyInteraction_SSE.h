/*
 *
 * bodybodyInteraction_SSE.h
 *
 * Intel x86/x86_64 SSE implementation of N-body computation.
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

#include <xmmintrin.h>

static inline __m128 
rcp_sqrt_nr_ps(const __m128 x)
{
    const __m128
        nr      = _mm_rsqrt_ps(x),
        muls    = _mm_mul_ps(_mm_mul_ps(nr, nr), x),
        beta    = _mm_mul_ps(_mm_set_ps1(0.5f), nr),
        gamma   = _mm_sub_ps(_mm_set_ps1(3.0f), muls);

    return _mm_mul_ps(beta, gamma);
}

static inline __m128
horizontal_sum_ps( const __m128 x )
{
    const __m128 t = _mm_add_ps(x, _mm_movehl_ps(x, x));
    return _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
}

inline void
bodyBodyInteraction(
    __m128& fx, 
    __m128& fy, 
    __m128& fz,
    
    const __m128& x0, 
    const __m128& y0, 
    const __m128& z0,
    
    const __m128& x1, 
    const __m128& y1, 
    const __m128& z1, 
    const __m128& mass1,

    const __m128& softeningSquared )
{
    // r_01  [3 FLOPS]
    __m128 dx = _mm_sub_ps( x1, x0 );
    __m128 dy = _mm_sub_ps( y1, y0 );
    __m128 dz = _mm_sub_ps( z1, z0 );

    // d^2 + e^2 [6 FLOPS]
    __m128 distSq = 
        _mm_add_ps( 
            _mm_add_ps( 
                _mm_mul_ps( dx, dx ), 
                _mm_mul_ps( dy, dy ) 
            ), 
            _mm_mul_ps( dz, dz ) 
        );
    distSq = _mm_add_ps( distSq, softeningSquared );

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    __m128 invDist = rcp_sqrt_nr_ps( distSq );
    __m128 invDistCube = 
        _mm_mul_ps( 
            invDist, 
            _mm_mul_ps( 
                invDist, invDist ) 
        );

    // s = m_j * invDistCube [1 FLOP]
    __m128 s = _mm_mul_ps( mass1, invDistCube );

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    fx = _mm_add_ps( fx, _mm_mul_ps( dx, s ) );
    fy = _mm_add_ps( fy, _mm_mul_ps( dy, s ) );
    fz = _mm_add_ps( fz, _mm_mul_ps( dz, s ) );
}

