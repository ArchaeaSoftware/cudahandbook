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

#include <chError.h>
#include <chTimer.h>
#include <chThread.h>

#include "nbody.h"
#include "bodybodyInteraction.cuh"

using namespace cudahandbook::threading;

__global__ void
ComputeNBodyGravitation_multiGPU( float *force, float *posMass, float softeningSquared, size_t base, size_t n, size_t N )
{
    extern __shared__ float4 shPosMass[];
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x;
              i < n;
              i += blockDim.x*gridDim.x )
    {
        float acc[3] = {0};
        size_t myIndex = base+i;
        float4 myPosMass = ((float4 *) posMass)[myIndex];

        for ( int j = 0; j < N; j += blockDim.x ) {
            shPosMass[threadIdx.x] = ((float4 *) posMass)[j+threadIdx.x];
            __syncthreads();
//#pragma unroll 32
            for ( size_t i = 0; i < blockDim.x; i++ ) {
                float fx, fy, fz;
                float4 bodyPosMass = shPosMass[i];

                bodyBodyInteraction( &fx, &fy, &fz, myPosMass.x, myPosMass.y, myPosMass.z, bodyPosMass.x, bodyPosMass.y, bodyPosMass.z, bodyPosMass.w, softeningSquared );
                acc[0] += fx;
                acc[1] += fy;
                acc[2] += fz;
            }
            __syncthreads();
        }
        force[3*myIndex+0] = acc[0];
        force[3*myIndex+1] = acc[1];
        force[3*myIndex+2] = acc[2];
    }
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
    float *dptrPosMass;
    float *dptrForce;

    //
    // Each GPU has its own device pointer to the host pointer.
    //
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrPosMass, p->hostPosMass, 0 ) );
    CUDART_CHECK( cudaHostGetDevicePointer( &dptrForce, p->hostForce, 0 ) );
    ComputeNBodyGravitation_multiGPU<<<300,256,256*sizeof(float4)>>>( 
        dptrForce,
        dptrPosMass,
        p->softeningSquared,
        p->i,
        p->n,
        p->N );
    CUDART_CHECK( cudaDeviceSynchronize() );
Error:
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
        size_t bodiesPerCore = N / g_numGPUs;
        if ( N % g_numThreads ) {
            return 0.0f;
        }

        size_t i;
        for ( i = 0; i < g_numGPUs; i++ ) {
            pgpu[i].hostPosMass = g_hostAOS_PosMass;
            pgpu[i].hostForce = g_hostAOS_Force;

            pgpu[i].softeningSquared = softeningSquared;

            pgpu[i].i = bodiesPerCore*i;
            pgpu[i].n = bodiesPerCore;
            pgpu[i].N = N;

            g_ThreadPool[i].delegateAsynchronous( gpuWorkerThread, &pgpu[i] );
        }
        workerThread::waitAll( g_ThreadPool, g_numGPUs );
        delete[] pgpu;
    }

    chTimerGetTime( &end );
    return chTimerElapsedTime( &start, &end ) * 1000.0f;
}
