/*
 *
 * bodybodyInteraction_NEON.h
 *
 * ARM NEON implementation of N-body computation.
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

#ifdef __ARM_NEON__

#include <arm_neon.h>

typedef float vf32x4_t __attribute__ ((vector_size(16),aligned(1)));

static const vf32x4_t vec_zero = {0.0f, 0.0f, 0.0f, 0.0f};

typedef union {
	float32x4_t v;
	float f[4];
	vf32x4_t p;
} v4;

static inline vf32x4_t
_vec_set_ps1(float f)
{
	v4 r;
	r.v = vdupq_n_f32(f);
	return r.p;
}

static inline float
_vec_sum(vf32x4_t const &v)
{
	float32x2_t r;
	v4 iv;
	iv.p = v;
	r = vadd_f32(vget_high_f32(iv.v), vget_low_f32(iv.v));
	return vget_lane_f32(vpadd_f32(r, r), 0);
}

static inline vf32x4_t
rcp_sqrt_nr_ps(const vf32x4_t& _v) {
	v4 vec, result;
	vec.p = _v;
	result.v = vrsqrteq_f32(vec.v);
	result.v = vmulq_f32(vrsqrtsq_f32(vmulq_f32(result.v, result.v), vec.v), result.v);
	return result.p;
}

inline void
bodyBodyInteraction(
    vf32x4_t& fx,
	vf32x4_t& fy,
    vf32x4_t& fz,

    const vf32x4_t& x0,
    const vf32x4_t& y0,
    const vf32x4_t& z0,

    const vf32x4_t& x1,
    const vf32x4_t& y1,
    const vf32x4_t& z1,
    const vf32x4_t& mass1,

    const vf32x4_t& softeningSquared )
{
    // r_01  [3 FLOPS]
    vf32x4_t dx = x1 - x0;
    vf32x4_t dy = y1 - y0;
    vf32x4_t dz = z1 - z0;

    // d^2 + e^2 [6 FLOPS]
    vf32x4_t distSq = ( dx * dx ) + ( dy * dy ) + ( dz * dz );
    distSq = distSq + softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    vf32x4_t invDist = rcp_sqrt_nr_ps ( distSq );
    vf32x4_t invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    vf32x4_t s = mass1 * invDistCube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
	fx = fx + (dx * s);
	fy = fy + (dy * s);
	fz = fz + (dz * s);
}

#endif
