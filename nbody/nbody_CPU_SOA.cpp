/*
 *
 * nbody_CPU_SOA.h
 *
 * Scalar CPU implementation of the O(N^2) N-body calculation.
 * This SOA (structure of arrays) formulation blazes the trail
 * for an SSE implementation.
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

#ifndef NO_CUDA
#define NO_CUDA
#endif
#include <chCUDA.h>
#include <chTimer.h>

#include "bodybodyInteraction.cuh"

float
ComputeGravitation_SOA(
    float *force[3],
    float *pos[4],
    float *mass,
    float softeningSquared,
    size_t N
)
{
    chTimerTimestamp start, end;
    memset( force[0], 0, N*sizeof(float) );
    memset( force[1], 0, N*sizeof(float) );
    memset( force[2], 0, N*sizeof(float) );
    chTimerGetTime( &start );
    for (size_t i = 0; i < N; i++)
    {
        float acc[3] = {0, 0, 0};
        float myX = pos[0][i];
        float myY = pos[1][i];
        float myZ = pos[2][i];

        for ( size_t j = 0; j < i; j++ ) {

            if ( j==i ) continue;

            float bodyX = pos[0][j];
            float bodyY = pos[1][j];
            float bodyZ = pos[2][j];
            float bodyMass = mass[j];

            float fx, fy, fz;
            bodyBodyInteraction<float>(
                &fx, &fy, &fz,
                myX, myY, myZ,
                bodyX, bodyY, bodyZ, bodyMass,
                softeningSquared );

            acc[0] += fx;
            acc[1] += fy;
            acc[2] += fz;

            force[0][j] += -fx;
            force[1][j] += -fy;
            force[2][j] += -fz;
        }

        force[0][i] += acc[0];
        force[1][i] += acc[1];
        force[2][i] += acc[2];
    }
    chTimerGetTime( &end );
    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}
