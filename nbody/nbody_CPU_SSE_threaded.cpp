/*
 *
 * nbody_CPU_SSE_threaded.cpp
 *
 * Multithreaded SSE CPU implementation of the O(N^2) N-body calculation.
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

#ifdef __SSE__

#include <xmmintrin.h>

#include <chTimer.h>
#include <chThread.h>

#include "nbody.h"
#include "bodybodyInteraction_SSE.h"
#include "nbody_CPU_SIMD.h"

using namespace cudahandbook::threading;

struct sseDelegation {
    size_t i;   // base offset for this thread to process
    size_t n;   // size of this thread's problem
    size_t N;   // total number of bodies

    float *hostPosSOA[3];
    float *hostMassSOA;
    float *hostForceSOA[3];
    float softeningSquared;

};

static void
sseWorkerThread( void *_p )
{
    sseDelegation *p = (sseDelegation *) _p;
    for (int k = 0; k < p->n; k++)
    {
        int i = p->i + k;
        __m128 ax = _mm_setzero_ps();
        __m128 ay = _mm_setzero_ps();
        __m128 az = _mm_setzero_ps();
        __m128 *px = (__m128 *) p->hostPosSOA[0];
        __m128 *py = (__m128 *) p->hostPosSOA[1];
        __m128 *pz = (__m128 *) p->hostPosSOA[2];
        __m128 *pmass = (__m128 *) p->hostMassSOA;
        __m128 x0 = _mm_set_ps1( p->hostPosSOA[0][i] );
        __m128 y0 = _mm_set_ps1( p->hostPosSOA[1][i] );
        __m128 z0 = _mm_set_ps1( p->hostPosSOA[2][i] );

        for ( int j = 0; j < p->N/4; j++ ) {
            
            bodyBodyInteraction( 
                ax, ay, az,
                x0, y0, z0, 
                px[j], py[j], pz[j], pmass[j], 
                _mm_set_ps1( p->softeningSquared ) );

        }
        // Accumulate sum of four floats in the SSE register
        ax = horizontal_sum_ps( ax );
        ay = horizontal_sum_ps( ay );
        az = horizontal_sum_ps( az );

        _mm_store_ss( (float *) &p->hostForceSOA[0][i], ax );
        _mm_store_ss( (float *) &p->hostForceSOA[1][i], ay );
        _mm_store_ss( (float *) &p->hostForceSOA[2][i], az );
    }
}

float
ComputeGravitation_SIMD_threaded(
    float *force[3], 
    float *pos[4],
    float *mass,
    float softeningSquared,
    size_t N
)
{
    chTimerTimestamp start, end;
    chTimerGetTime( &start );

    {
        sseDelegation *psse = new sseDelegation[g_numCPUCores];
        size_t bodiesPerCore = N / g_numCPUCores;
        for ( size_t i = 0; i < g_numCPUCores; i++ ) {
            psse[i].hostPosSOA[0] = pos[0];
            psse[i].hostPosSOA[1] = pos[1];
            psse[i].hostPosSOA[2] = pos[2];
            psse[i].hostMassSOA = mass;
            psse[i].hostForceSOA[0] = force[0];
            psse[i].hostForceSOA[1] = force[1];
            psse[i].hostForceSOA[2] = force[2];
            psse[i].softeningSquared = softeningSquared;

            psse[i].i = bodiesPerCore*i;
            psse[i].n = bodiesPerCore;
            psse[i].N = N;

            g_CPUThreadPool[i].delegateAsynchronous( 
                sseWorkerThread, 
                &psse[i] );
        }
        workerThread::waitAll( g_CPUThreadPool, g_numCPUCores );
        delete[] psse;
    }

    chTimerGetTime( &end );

    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}

#endif
