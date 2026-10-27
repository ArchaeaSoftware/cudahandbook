/*
 *
 * nbody_multiGPU_buffers.cu
 *
 * Persistent device buffers shared by the multi-GPU N-body implementations.
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

#include <stdio.h>

#include <chError.h>

#include "nbody_multiGPU_buffers.h"

multiGPUBuffers g_multiGPUBuffers[g_maxGPUs];

static int    g_cBuffersAllocated = 0;
static size_t g_NBuffersAllocated = 0;

void
multiGPUBuffersRelease( void )
{
    int oldDevice;
    if ( cudaSuccess != cudaGetDevice( &oldDevice ) ) {
        return;
    }
    for ( int i = 0; i < g_cBuffersAllocated; i++ ) {
        cudaSetDevice( i );
        cudaFree( g_multiGPUBuffers[i].posMass );
        cudaFree( g_multiGPUBuffers[i].force );
        g_multiGPUBuffers[i].posMass = NULL;
        g_multiGPUBuffers[i].force = NULL;
    }
    cudaSetDevice( oldDevice );
    g_cBuffersAllocated = 0;
    g_NBuffersAllocated = 0;
}

bool
multiGPUBuffersAcquire( int numGPUs, size_t N )
{
    cudaError_t status;
    int oldDevice;
    bool bRet = false;

    if ( numGPUs > g_maxGPUs ) {
        return false;
    }
    if ( (numGPUs == g_cBuffersAllocated) && (N == g_NBuffersAllocated) ) {
        return true;
    }
    multiGPUBuffersRelease();

    cuda(GetDevice( &oldDevice ) );
    for ( int i = 0; i < numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        // Count the device in before allocating for it, so that a failure
        // partway through releases what this loop already allocated rather
        // than leaking it.  Release frees a NULL pointer harmlessly.
        g_cBuffersAllocated = i+1;
        cuda(Malloc( &g_multiGPUBuffers[i].posMass, 4*N*sizeof(float) ) );
        cuda(Malloc( &g_multiGPUBuffers[i].force, 3*N*sizeof(float) ) );
    }
    g_NBuffersAllocated = N;
    bRet = true;

Error:
    cudaSetDevice( oldDevice );
    if ( ! bRet ) {
        multiGPUBuffersRelease();
    }
    return bRet;
}
