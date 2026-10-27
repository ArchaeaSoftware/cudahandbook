/*
 *
 * nbody_multiGPU_buffers.h
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

#ifndef __CUDAHANDBOOK_NBODY_MULTIGPU_BUFFERS_H__
#define __CUDAHANDBOOK_NBODY_MULTIGPU_BUFFERS_H__

#include "nbody.h"

// Each multi-GPU implementation needs, on every device, room for all N
// bodies' positions and masses and room for the forces it computes.  These
// buffers used to be allocated and freed inside the timed region, once per
// device per timestep.  That charged every timestep for a cudaMalloc whose
// cost rises with the device count while the per-device workload falls, which
// is the one bias a scaling study cannot tolerate: it turns a measurement of
// communication and compute into a measurement of the allocator.
//
// Acquire them before the clock starts instead.  The force buffer holds all
// 3*N floats rather than the 3*N/numGPUs a device strictly needs, because the
// GPU cross-check has every device compute every body.

struct multiGPUBuffers {
    float *posMass;     // 4*N floats, on device
    float *force;       // 3*N floats, on device
};

extern multiGPUBuffers g_multiGPUBuffers[g_maxGPUs];

// Allocates the buffers for devices [0, numGPUs) if they are not already
// allocated at this geometry, and reallocates them if N or numGPUs changed.
// Leaves the current device as it found it.  Returns false if any allocation
// failed, in which case nothing is left allocated.
bool multiGPUBuffersAcquire( int numGPUs, size_t N );

// Frees every allocation.  Safe to call when nothing is allocated.
void multiGPUBuffersRelease( void );

#endif
