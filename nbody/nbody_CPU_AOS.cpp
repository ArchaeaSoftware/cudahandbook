/*
 *
 * nbody_CPU_AOS.h
 *
 * Scalar CPU implementation of the O(N^2) N-body calculation.
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

#ifndef NO_CUDA
#define NO_CUDA
#endif
#include <chCUDA.h>
#include <chrono>

#include "bodybodyInteraction.cuh"

float
ComputeGravitation_AOS(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    for ( size_t i = 0; i < N; i++ )
    {
        float acc[3] = {0, 0, 0};
        float myX = posMass[i*4+0];
        float myY = posMass[i*4+1];
        float myZ = posMass[i*4+2];

        for ( size_t j = 0; j < N; j++ ) {
            float fx, fy, fz;
            float bodyX = posMass[j*4+0];
            float bodyY = posMass[j*4+1];
            float bodyZ = posMass[j*4+2];
            float bodyMass = posMass[j*4+3];

            bodyBodyInteraction<float>(
                &fx, &fy, &fz,
                myX, myY, myZ,
                bodyX, bodyY, bodyZ, bodyMass,
                softeningSquared );
            acc[0] += fx;
            acc[1] += fy;
            acc[2] += fz;
        }

        force[3*i+0] = acc[0];
        force[3*i+1] = acc[1];
        force[3*i+2] = acc[2];
    }
    end = std::chrono::steady_clock::now();
    return (float) std::chrono::duration<double>(end - start).count() * 1000.0f;
}
