/*
 *
 * bodybodyInteraction_AltiVec.h
 *
 * PowerPC AltiVec implementation of N-body computation.
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

#ifdef __ALTIVEC__
#include <altivec.h>

typedef vector float v4sf;

static const v4sf vec_zero = {0.0f, 0.0f, 0.0f, 0.0f};

typedef union {
	float f[4];
	v4sf p;
} v4;

#ifndef __VSX__
static inline v4sf
vec_mul(v4sf x, v4sf y)
{
	return vec_madd(x, y, vec_zero);
}
#endif

static inline v4sf
_vec_set_ps1(float f)
{
	v4 r;
	r.f[0] = f;
	r.f[1] = f;
	r.f[2] = f;
	r.f[3] = f;
	return r.p;
}

static inline float
_vec_sum(v4sf v)
{
	v4 r;
	r.p = v;
	return r.f[0] + r.f[1] + r.f[2] + r.f[3];
}

static inline v4sf
rcp_sqrt_nr_ps(const v4sf x)
{
    const v4sf
        nr      = vec_rsqrte(x),
		muls    = vec_mul(vec_mul(nr, nr), x),
        beta    = vec_mul(_vec_set_ps1(0.5f), nr),
        gamma   = vec_sub(_vec_set_ps1(3.0f), muls);

	return vec_mul(beta, gamma);
}

inline void
bodyBodyInteraction(
    v4sf& fx,
	v4sf& fy,
    v4sf& fz,

    const v4sf& x0,
    const v4sf& y0,
    const v4sf& z0,

    const v4sf& x1,
    const v4sf& y1,
    const v4sf& z1,
    const v4sf& mass1,

    const v4sf& softeningSquared )
{
    // r_01  [3 FLOPS]
    v4sf dx = vec_sub( x1, x0 );
    v4sf dy = vec_sub( y1, y0 );
    v4sf dz = vec_sub( z1, z0 );

    // d^2 + e^2 [6 FLOPS]
    v4sf distSq =
        vec_add(
            vec_add(
                vec_mul( dx, dx ),
                vec_mul( dy, dy )
            ),
            vec_mul( dz, dz )
        );
    distSq = vec_add( distSq, softeningSquared );

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    v4sf invDist = rcp_sqrt_nr_ps( distSq );
    v4sf invDistCube =
        vec_mul(
            invDist,
            vec_mul(
                invDist, invDist )
        );

    // s = m_j * invDistCube [1 FLOP]
    v4sf s = vec_mul( mass1, invDistCube );

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    fx = vec_add( fx, vec_mul( dx, s ) );
    fy = vec_add( fy, vec_mul( dy, s ) );
    fz = vec_add( fz, vec_mul( dz, s ) );
}

#endif
