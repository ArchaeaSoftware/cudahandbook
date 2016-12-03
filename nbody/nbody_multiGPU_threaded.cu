/*
 *
 * nbody_multiGPU_threaded.cu
 *
 * Multithreaded multi-GPU implementation of the O(N^2) N-body calculation.
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

#include <stdio.h>

#include <chError.h>
#include <chTimer.h>
#include <chThread.h>

#include "nbody.h"
#include "bodybodyInteraction.cuh"
#include "nbody_multiGPU_shared.cuh"

using namespace cudahandbook::threading;

__global__ void
ComputeNBodyGravitation_multiGPU( 
    float *force, 
    float *posMass, 
    float softeningSquared, 
    size_t base, 
    size_t n, 
    size_t N )
{
    ComputeNBodyGravitation_Shared_multiGPU( 
        force, 
        posMass, 
        softeningSquared, 
        base, 
        n,
        N );
}

struct gpuDelegation {
    size_t i;   // base offset for this thread to process
    size_t n;   // size of this thread's problem
    size_t N;   // total number of bodies

    float *hostPosMass;
    float *hostForce;
    float softeningSquared;

    cudaError_t status;
};

void
gpuWorkerThread( void *_p )
{
    cudaError_t status;
    gpuDelegation *p = (gpuDelegation *) _p;
    float *dptrPosMass = 0;
    float *dptrForce = 0;

    //
    // Each GPU has its own device pointer to the host pointer.
    //
    cuda(Malloc( &dptrPosMass, 4*p->N*sizeof(float) ) );
    cuda(Malloc( &dptrForce, 3*p->n*sizeof(float) ) );
    cuda(MemcpyAsync( 
        dptrPosMass, 
        p->hostPosMass, 
        4*p->N*sizeof(float), 
        cudaMemcpyHostToDevice ) );
    ComputeNBodyGravitation_multiGPU<<<300,256,256*sizeof(float4)>>>( 
        dptrForce,
        dptrPosMass,
        p->softeningSquared,
        p->i,
        p->n,
        p->N );
    // NOTE: synchronous memcpy, so no need for further 
    // synchronization with device
    cuda(Memcpy( 
        p->hostForce+3*p->i, 
        dptrForce, 
        3*p->n*sizeof(float), 
        cudaMemcpyDeviceToHost ) );
Error:
    cudaFree( dptrPosMass );
    cudaFree( dptrForce );
    p->status = status;
}

float
ComputeGravitation_multiGPU_threaded( 
    float *force, 
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    chTimerTimestamp start, end;
    chTimerGetTime( &start );
    {
        gpuDelegation *pgpu = new gpuDelegation[g_numGPUs];
        size_t bodiesPerGPU = N / g_numGPUs;
        if ( N % g_numGPUs ) {
            return 0.0f;
        }

        size_t i;
        for ( i = 0; i < g_numGPUs; i++ ) {
            pgpu[i].hostPosMass = g_hostAOS_PosMass;
            pgpu[i].hostForce = g_hostAOS_Force;

            pgpu[i].softeningSquared = softeningSquared;

            pgpu[i].i = bodiesPerGPU*i;
            pgpu[i].n = bodiesPerGPU;
            pgpu[i].N = N;

            g_GPUThreadPool[i].delegateAsynchronous( 
                gpuWorkerThread, 
                &pgpu[i] );
        }
        workerThread::waitAll( g_GPUThreadPool, g_numGPUs );
        delete[] pgpu;
    }

    chTimerGetTime( &end );
    return chTimerElapsedTime( &start, &end ) * 1000.0f;
}
