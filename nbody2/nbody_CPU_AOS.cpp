/*
 *
 * nbody_CPU_AOS.h
 *
 * Scalar CPU implementation of the O(N^2) N-body calculation.
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

#include "nbody.h"
#include "bodybodyInteraction.cuh"

template<typename T>
bool
NBodyAlgorithm<T>::Initialize( size_t N, T softening )
{
    N_ = N;
    softening_ = softening;
    return true;
}

template<typename T>
float
NBodyAlgorithm<T>::computeTimeStep( std::vector<Force3D<T> >& force )
{
    T softeningSquared = softening_*softening_;
    chTimerTimestamp start, end;
    chTimerGetTime( &start );
    for ( size_t i = 0; i < N; i++ )
    {
        Force3D<T> acc = { 0, 0, 0 };
        float myX = posMass_[i].x_;
        float myY = posMass_[i].y_;
        float myZ = posMass_[i].z_;

        for ( size_t j = 0; j < N; j++ ) {
            float fx, fy, fz;
            float bodyX = posMass_[j].x_;
            float bodyY = posMass_[j].y_;
            float bodyZ = posMass_[j].z_;
            float bodyMass = posMass_[j].mass_;

            bodyBodyInteraction<float>(
                &fx, &fy, &fz,
                myX, myY, myZ,
                bodyX, bodyY, bodyZ, bodyMass,
                softeningSquared );
            acc.dx_ += fx;
            acc.dy_ += fy;
            acc.dz_ += fz;
        }

        force[i] = acc.dx_;
        force[i] = acc.dy_;
        force[i] = acc.dz_;
    }
    chTimerGetTime( &end );
    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
    
}

float
ComputeGravitation_AOS(
    float *force,
    const std::vector<PosMass<float>>& posMass,// float *posMass,
    float softeningSquared,
    size_t N
)
{
    chTimerTimestamp start, end;
    chTimerGetTime( &start );
    for ( size_t i = 0; i < N; i++ )
    {
        float acc[3] = {0, 0, 0};
        float myX = posMass[i].x_;
        float myY = posMass[i].y_;
        float myZ = posMass[i].z_;

        for ( size_t j = 0; j < N; j++ ) {
            float fx, fy, fz;
            float bodyX = posMass[j].x_;
            float bodyY = posMass[j].y_;
            float bodyZ = posMass[j].z_;
            float bodyMass = posMass[j].mass_;

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
    chTimerGetTime( &end );
    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}
