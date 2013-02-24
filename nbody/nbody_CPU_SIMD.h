/*
 *
 * nbody_CPU_SSE.h
 *
 * SSE CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for SSE.
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

#if defined(HAVE_SSE)

#define HAVE_SIMD
#define HAVE_SIMD_THREADED
#ifdef USE_OPENMP
#define HAVE_SIMD_OPENMP
#endif

#include "nbody_CPU_SSE.h"
#include "nbody_CPU_SSE_threaded.h"
#ifdef USE_OPENMP
#include "nbody_CPU_SSE_openmp.h"
#endif

#elif defined(HAVE_ALTIVEC)

#define HAVE_SIMD
#ifdef USE_OPENMP
#define HAVE_SIMD_OPENMP
#endif

#include "nbody_CPU_AltiVec.h"
#ifdef USE_OPENMP
#include "nbody_CPU_AltiVec_openmp.h"
#endif

#endif

float
ComputeGravitation_SIMD(
    float *force[3],
    float *pos[4],
    float *mass,
    float softeningSquared,
    size_t N
);

float
ComputeGravitation_SIMD_threaded(
    float *force[3],
    float *pos[4],
    float *mass,
    float softeningSquared,
    size_t N
);

float
ComputeGravitation_SIMD_openmp(
    float *force[3],
    float *pos[4],
    float *mass,
    float softeningSquared,
    size_t N
);
