/*
 *
 * nbody_multiGPU.cu
 *
 * Single-threaded multi-GPU implementation of the O(N^2) N-body calculation.
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
#include "nbody_GPU_shared.cuh"

#include "bodybodyInteraction.cuh"

using namespace cudahandbook::threading;

__global__ void
ComputeNBodyGravitation_multiGPU_onethread( 
    float *force, 
    float *posMass, 
    float softeningSquared, 
    size_t base, 
    size_t n, 
    size_t N )
{
    ComputeNBodyGravitation_Shared_device( 
        force, 
        posMass, 
        softeningSquared, 
        base, 
        N );
}

float
ComputeGravitation_multiGPU_singlethread( 
    float *force, 
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;

    float ret = 0.0f;
    
    float *dptrPosMass = 0;
    float *dptrForce = 0;

    chTimerTimestamp start, end;
    chTimerGetTime( &start );
    
    size_t bodiesPerGPU = N / g_numGPUs;
    if ( N % g_numGPUs ) {
        return 0.0f;
    }
    
    // kick off the asynchronous memcpy's - overlap GPUs pulling
    // host memory with the CPU time needed to do the memory 
    // allocations.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        CUDART_CHECK( cudaSetDevice( i ) );
        CUDART_CHECK( cudaMalloc( &dptrPosMass, 4*N*sizeof(float) ) );
        CUDART_CHECK( cudaMalloc( &dptrForce, 3*bodiesPerGPU*sizeof(float) ) );
        CUDART_CHECK( cudaMemcpyAsync( 
            dptrPosMass, 
            g_hostAOS_PosMass, 
            4*N*sizeof(float), 
            cudaMemcpyHostToDevice ) );
    }
    for ( int i = 0; i < g_numGPUs; i++ ) {
        CUDART_CHECK( cudaSetDevice( i ) );
        ComputeNBodyGravitation_multiGPU_onethread<<<300,256,256*sizeof(float4)>>>( 
            dptrForce,
            dptrPosMass,
            softeningSquared,
            i*bodiesPerGPU,
            bodiesPerGPU,
            N );
        CUDART_CHECK( cudaMemcpyAsync( 
            g_hostAOS_Force+3*bodiesPerGPU*i, 
            dptrForce, 
            3*bodiesPerGPU*sizeof(float), 
            cudaMemcpyDeviceToHost ) );
    }
    // Synchronize with each GPU in turn.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        CUDART_CHECK( cudaSetDevice( i ) );
        CUDART_CHECK( cudaDeviceSynchronize() );
    }
    chTimerGetTime( &end );
    return chTimerElapsedTime( &start, &end ) * 1000.0f;
Error:
    cudaFree( dptrPosMass );
    cudaFree( dptrForce );
    return ret;
}
