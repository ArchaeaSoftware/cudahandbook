/*
 *
 * nbody_CPU_AltiVec.cpp
 *
 * Multithreaded AltiVec CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for AltiVec.
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
#ifdef _OPENMP
#include <chTimer.h>

#include "nbody.h"
#include "bodybodyInteraction_AltiVec.h"
#include "nbody_CPU_SSE.h"

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
    for (int i = 0; i < N; i++)
    {
        v4sf ax = vec_zero;
        v4sf ay = vec_zero;
        v4sf az = vec_zero;
        v4sf *px = (v4sf *) pos[0];
        v4sf *py = (v4sf *) pos[1];
        v4sf *pz = (v4sf *) pos[2];
        v4sf *pmass = (v4sf *) mass;
        v4sf x0 = _vec_set_ps1( pos[0][i] );
        v4sf y0 = _vec_set_ps1( pos[1][i] );
        v4sf z0 = _vec_set_ps1( pos[2][i] );

        for ( int j = 0; j < N/4; j++ ) {

            bodyBodyInteraction(
                ax, ay, az,
                x0, y0, z0,
                px[j], py[j], pz[j], pmass[j],
                _vec_set_ps1( softeningSquared ) );

        }

        // Accumulate sum of four floats in the AltiVec register
        force[0][i] = _vec_sum( ax );
        force[1][i] = _vec_sum( ay );
        force[2][i] = _vec_sum( az );
    }

    chTimerGetTime( &end );

    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}
#endif
#endif
