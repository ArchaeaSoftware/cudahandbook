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
#include "nbody_multiGPU_shared.cuh"

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
    ComputeNBodyGravitation_Shared_multiGPU( 
        force, 
        posMass, 
        softeningSquared, 
        base, 
        n,
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

    float *dptrPosMass[g_maxGPUs];
    float *dptrForce[g_maxGPUs];
    int oldDevice;

    chTimerTimestamp start, end;
    chTimerGetTime( &start );

    memset( dptrPosMass, 0, sizeof(dptrPosMass) );
    memset( dptrForce, 0, sizeof(dptrForce) );
    size_t bodiesPerGPU = N / g_numGPUs;
    if ( (0 != N % g_numGPUs) || (g_numGPUs > g_maxGPUs) ) {
        return 0.0f;
    }
    cuda(GetDevice( &oldDevice ) );

    // kick off the asynchronous memcpy's - overlap GPUs pulling
    // host memory with the CPU time needed to do the memory 
    // allocations.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        cuda(Malloc( &dptrPosMass[i], 4*N*sizeof(float) ) );
        // we only need 3*N floatsw for the cross-check. otherwise we 
        // would need 3*bodiesPerGPU
        cuda(Malloc( &dptrForce[i], 3*N*sizeof(float) ) );
        cuda(MemcpyAsync( 
            dptrPosMass[i], 
            g_hostAOS_PosMass, 
            4*N*sizeof(float), 
            cudaMemcpyHostToDevice ) );
    }
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        if ( g_bGPUCrossCheck ) {
            ComputeNBodyGravitation_multiGPU_onethread<<<300,256,256*sizeof(float4)>>>( 
                dptrForce[i],
                dptrPosMass[i],
                softeningSquared,
                0,
                N,
                N );
            cuda(MemcpyAsync( 
                g_hostAOS_gpuCrossCheckForce[i], 
                dptrForce[i], 
                3*N*sizeof(float), 
                cudaMemcpyDeviceToHost ) );
            cuda(MemcpyAsync( 
                g_hostAOS_Force+3*bodiesPerGPU*i, 
                dptrForce[i]+3*bodiesPerGPU*i, 
                3*bodiesPerGPU*sizeof(float), 
                cudaMemcpyDeviceToHost ) );
        }
        else {
            ComputeNBodyGravitation_multiGPU_onethread<<<300,256,256*sizeof(float4)>>>( 
                dptrForce[i],
                dptrPosMass[i],
                softeningSquared,
                i*bodiesPerGPU,
                bodiesPerGPU,
                N );
            cuda(MemcpyAsync( 
                g_hostAOS_Force+3*bodiesPerGPU*i, 
                dptrForce[i], 
                3*bodiesPerGPU*sizeof(float), 
                cudaMemcpyDeviceToHost ) );
        }
    }
    // Synchronize with each GPU in turn.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        cuda(DeviceSynchronize() );
    }
    chTimerGetTime( &end );
    ret = chTimerElapsedTime( &start, &end ) * 1000.0f;

    if ( g_fGPUCrosscheckOutput ) {
        if ( 1 != fwrite( g_hostAOS_Force, 3*N*sizeof(float), 1, g_fGPUCrosscheckOutput ) )
            goto Error;
    }
    if ( g_fGPUCrosscheckInput ) {
        if ( 1 != fread( g_hostAOS_Force_Golden, 3*N*sizeof(float), 1, g_fGPUCrosscheckInput ) )
            goto Error;
    }


Error:
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cudaFree( dptrPosMass[i] );
        cudaFree( dptrForce[i] );
    }
    cudaSetDevice( oldDevice );
    return ret;
}
