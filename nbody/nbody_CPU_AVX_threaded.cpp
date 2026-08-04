/*
 *
 * nbody_CPU_AVX_threaded.cpp
 *
 * Multithreaded AVX CPU implementation of the O(N^2) N-body calculation.
 * Uses SOA (structure of arrays) representation because it is a much
 * better fit for AVX.
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

#ifdef __AVX__

#include <immintrin.h>

#include <chrono>

#include <thread>
#include <vector>

#include "nbody.h"
#include "bodybodyInteraction_AVX.h"
#include "nbody_CPU_SIMD.h"

struct avxDelegation {
    size_t i;   // base offset for this thread to process
    size_t n;   // size of this thread's problem
    size_t N;   // total number of bodies

    float *hostPosSOA[3];
    float *hostMassSOA;
    float *hostForceSOA[3];
    float softeningSquared;

};

static void
avxWorkerThread( void *_p )
{
    avxDelegation *p = (avxDelegation *) _p;
    const __m256 softening = _mm256_set1_ps( p->softeningSquared );
    for (int k = 0; k < p->n; k++)
    {
        int i = p->i + k;
        __m256 ax = _mm256_setzero_ps();
        __m256 ay = _mm256_setzero_ps();
        __m256 az = _mm256_setzero_ps();
        __m256 x0 = _mm256_set1_ps( p->hostPosSOA[0][i] );
        __m256 y0 = _mm256_set1_ps( p->hostPosSOA[1][i] );
        __m256 z0 = _mm256_set1_ps( p->hostPosSOA[2][i] );

        for ( int j = 0; j < p->N/8; j++ ) {

            bodyBodyInteraction(
                ax, ay, az,
                x0, y0, z0,
                _mm256_loadu_ps( p->hostPosSOA[0] + 8*j ),
                _mm256_loadu_ps( p->hostPosSOA[1] + 8*j ),
                _mm256_loadu_ps( p->hostPosSOA[2] + 8*j ),
                _mm256_loadu_ps( p->hostMassSOA   + 8*j ),
                softening );

        }
        // Sum the eight partial forces accumulated in each YMM register
        _mm_store_ss( &p->hostForceSOA[0][i], horizontal_sum_ps( ax ) );
        _mm_store_ss( &p->hostForceSOA[1][i], horizontal_sum_ps( ay ) );
        _mm_store_ss( &p->hostForceSOA[2][i], horizontal_sum_ps( az ) );
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
    // AVX processes eight bodies at a time; refuse a body count that is not
    // a multiple of eight rather than silently dropping the remainder.
    requireBodyCountForAVX( N );

    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();

    {
        avxDelegation *pavx = new avxDelegation[g_numCPUCores];
        std::vector<std::thread> threads;
        for ( size_t i = 0; i < g_numCPUCores; i++ ) {
            pavx[i].hostPosSOA[0] = pos[0];
            pavx[i].hostPosSOA[1] = pos[1];
            pavx[i].hostPosSOA[2] = pos[2];
            pavx[i].hostMassSOA = mass;
            pavx[i].hostForceSOA[0] = force[0];
            pavx[i].hostForceSOA[1] = force[1];
            pavx[i].hostForceSOA[2] = force[2];
            pavx[i].softeningSquared = softeningSquared;

            // Divide the bodies as evenly as possible among the cores. This
            // split is independent of the AVX width: a core may be handed any
            // number of bodies, so an uneven division just gives some cores one
            // more body than others rather than dropping the remainder.
            size_t begin = N *  i      / g_numCPUCores;
            size_t end   = N * (i + 1) / g_numCPUCores;
            pavx[i].i = begin;
            pavx[i].n = end - begin;
            pavx[i].N = N;

            threads.emplace_back( avxWorkerThread, &pavx[i] );
        }
        for ( auto &t : threads ) t.join();
        delete[] pavx;
    }

    end = std::chrono::steady_clock::now();

    return (float) std::chrono::duration<double>(end - start).count() * 1000.0f;
}

#endif
