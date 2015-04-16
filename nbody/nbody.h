/*
 *
 * nbody.h
 *
 * Header file to declare globals in nbody.cu
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

#ifndef __CUDAHANDBOOK_NBODY_H__
#define __CUDAHANDBOOK_NBODY_H__

#include "nbody_CPU_SIMD.h"

#include <chThread.h>

extern bool g_bCUDAPresent;
extern bool g_bGPUCrossCheck;

extern bool g_GPUCrosscheck;
#define NBODY_GOLDENFILE_VERSION 0x100
extern FILE *g_fGPUCrosscheckInput;
extern FILE *g_fGPUCrosscheckOutput;

extern float *g_hostAOS_PosMass;
extern float *g_hostAOS_VelInvMass;
extern float *g_hostAOS_Force;

// for GPU cross-check
const int g_maxGPUs = 32;
extern float *g_hostAOS_gpuCrossCheckForce[g_maxGPUs];

extern float *g_dptrAOS_PosMass;
extern float *g_dptrAOS_Force;


// Buffer to hold the golden version of the forces, used for comparison
// Along with timing results, we report the maximum relative error with 
// respect to this array.
extern float *g_hostAOS_Force_Golden;

extern float *g_hostSOA_Pos[3];
extern float *g_hostSOA_Force[3];
extern float *g_hostSOA_Mass;
extern float *g_hostSOA_InvMass;

extern size_t g_N;

extern float g_softening;
extern float g_damping;
extern float g_dt;

enum nbodyAlgorithm_enum {
    CPU_AOS = 0,    /* This is the golden implementation */
    CPU_AOS_tiled,
    CPU_SOA,
#ifdef HAVE_SIMD
    CPU_SIMD,
#endif
#ifdef HAVE_SIMD_THREADED
    CPU_SIMD_threaded,
#endif
#ifdef HAVE_SIMD_OPENMP
    CPU_SIMD_openmp,
#endif
    GPU_AOS,
    GPU_Shared,
    GPU_Const,
    multiGPU_SingleCPUThread,
    multiGPU_MultiCPUThread,
// SM 3.0 only
    GPU_Shuffle,
    GPU_AOS_tiled,
    GPU_AOS_tiled_const,
//    GPU_Atomic
};


static const char *rgszAlgorithmNames[] = { 
    "CPU_AOS", 
    "CPU_AOS_tiled", 
    "CPU_SOA", 
#ifdef HAVE_SIMD
    "CPU_SIMD",
#endif
#ifdef HAVE_SIMD_THREADED
    "CPU_SIMD_threaded",
#endif
#ifdef HAVE_SIMD_OPENMP
    "CPU_SIMD_openmp",
#endif
    "GPU_AOS", 
    "GPU_Shared", 
    "GPU_Const",
    "multiGPU_SingleCPUThread",
    "multiGPU_MultiCPUThread",
// SM 3.0 only
    "GPU_Shuffle",
    "GPU_AOS_tiled",
    "GPU_AOS_tiled_const",
//    "GPU_Atomic"
};

extern const char *rgszAlgorithmNames[];

extern enum nbodyAlgorithm_enum g_Algorithm;

//
// g_maxAlgorithm is used to determine when to rotate g_Algorithm back to CPU_AOS
// If CUDA is present, it is CPU_SIMD_threaded, otherwise GPU_Shuffle
// The CPU and GPU algorithms must be contiguous, and the logic in main() to
// initialize this value must be modified if any new algorithms are added.
//
extern enum nbodyAlgorithm_enum g_maxAlgorithm;
extern bool g_bCrossCheck;
extern bool g_bNoCPU;

extern cudahandbook::threading::workerThread *g_CPUThreadPool;
extern int g_numCPUCores;

extern int g_numGPUs;
extern cudahandbook::threading::workerThread *g_GPUThreadPool;

extern float ComputeGravitation_GPU_Shared           ( float *force, float *posMass, float softeningSquared, size_t N );
extern float ComputeGravitation_multiGPU_singlethread( float *force, float *posMass, float softeningSquared, size_t N );
extern float ComputeGravitation_multiGPU_threaded    ( float *force, float *posMass, float softeningSquared, size_t N );


#endif
