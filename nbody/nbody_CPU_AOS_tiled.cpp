/*
 *
 * nbody_CPU_AOS_tiled.h
 *
 * Scalar CPU implementation of the O(N^2) N-body calculation.
 * Performs the computation in 32x32 tiles.
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

template<int nTile>
void
DoDiagonalTile(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    for ( size_t _i = 0; _i < nTile; _i++ )
    {
        size_t i = iTile*nTile+_i;
        float acc[3] = {0, 0, 0};
        float myX = posMass[i*4+0];
        float myY = posMass[i*4+1];
        float myZ = posMass[i*4+2];

        for ( size_t _j = 0; _j < nTile; _j++ ) {
            size_t j = jTile*nTile+_j;

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

        force[3*i+0] += acc[0];
        force[3*i+1] += acc[1];
        force[3*i+2] += acc[2];
    }
}

template<int nTile>
void
DoNondiagonalTile(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    float symmetricX[nTile];
    float symmetricY[nTile];
    float symmetricZ[nTile];

    memset( symmetricX, 0, sizeof(symmetricX) );
    memset( symmetricY, 0, sizeof(symmetricY) );
    memset( symmetricZ, 0, sizeof(symmetricZ) );

    for ( size_t _i = 0; _i < nTile; _i++ )
    {
        size_t i = iTile*nTile+_i;
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        float myX = posMass[i*4+0];
        float myY = posMass[i*4+1];
        float myZ = posMass[i*4+2];

        for ( size_t _j = 0; _j < nTile; _j++ ) {
            size_t j = jTile*nTile+_j;

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

            ax += fx;
            ay += fy;
            az += fz;

            symmetricX[_j] -= fx;
            symmetricY[_j] -= fy;
            symmetricZ[_j] -= fz;

        }

        force[3*i+0] += ax;
        force[3*i+1] += ay;
        force[3*i+2] += az;

    }

    for ( size_t _j = 0; _j < nTile; _j++ ) {
        size_t j = jTile*nTile+_j;
        force[3*j+0] += symmetricX[_j];
        force[3*j+1] += symmetricY[_j];
        force[3*j+2] += symmetricZ[_j];
    }
}

template<int nTile>
float
ComputeGravitation_AOS_tiled(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    memset( force, 0, 3*N*sizeof(float) );
    chTimerTimestamp start, end;
    chTimerGetTime( &start );
    for ( size_t iTile = 0; iTile < N/nTile; iTile++ ) {
        for ( size_t jTile = 0; jTile <= iTile; jTile++ ) {
            if ( iTile != jTile ) {
                DoNondiagonalTile<nTile>(
                    force,
                    posMass,
                    softeningSquared,
                    iTile, jTile );
            }
        }
    }
    for ( size_t iTile = 0; iTile < N/nTile; iTile++ ) {
        DoDiagonalTile<nTile>(
            force,
            posMass,
            softeningSquared,
            iTile, iTile );
    }
    chTimerGetTime( &end );
    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}

float
ComputeGravitation_AOS_tiled(
    float *force,
    float *posMass,
    float softeningSquared,
    size_t N )
{
    return ComputeGravitation_AOS_tiled<32>(
        force,
        posMass,
        softeningSquared,
        N );
}
