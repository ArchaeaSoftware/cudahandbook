/*
 *
 * nbody_CPU_NEON.cpp
 *
 * Multithreaded NEON CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for NEON.
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
#ifdef _OPENMP
#include <chTimer.h>

#include "nbody.h"
#include "bodybodyInteraction_NEON.h"
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
    chTimerTimestamp start, end;
    chTimerGetTime( &start );

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        vf32x4_t ax = vec_zero;
        vf32x4_t ay = vec_zero;
        vf32x4_t az = vec_zero;
        vf32x4_t *px = (vf32x4_t *) pos[0];
        vf32x4_t *py = (vf32x4_t *) pos[1];
        vf32x4_t *pz = (vf32x4_t *) pos[2];
        vf32x4_t *pmass = (vf32x4_t *) mass;
        vf32x4_t x0 = _vec_set_ps1( pos[0][i] );
        vf32x4_t y0 = _vec_set_ps1( pos[1][i] );
        vf32x4_t z0 = _vec_set_ps1( pos[2][i] );

        for ( size_t j = 0; j < N/4; j++ ) {

            bodyBodyInteraction(
                ax, ay, az,
                x0, y0, z0,
                px[j], py[j], pz[j], pmass[j],
                _vec_set_ps1( softeningSquared ) );

        }

        // Accumulate sum of four floats in the NEON register
        force[0][i] = _vec_sum( ax );
        force[1][i] = _vec_sum( ay );
        force[2][i] = _vec_sum( az );
    }

    chTimerGetTime( &end );

    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}
#endif
#endif
