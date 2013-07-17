/*
 *
 * chCUDA.h
 *
 * Either loads CUDA or the dummy API interface, depending on build
 * requirements.
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


#ifndef __CHCUDA_H__
#define __CHCUDA_H__

#ifndef NO_CUDA

#ifndef __cuda_drvapi_dynlink_h__
#include <cuda.h>
#endif

#else

#include <stddef.h>
#include <math.h>
#include <memory.h>

#define __global__
#define __host__
#define __device__

typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;

static inline cudaError_t cudaGetDeviceCount( int *p )
{
    if (!p)
        return 1;
    *p = 0;
    return 0;
}

template<typename T>
static inline cudaError_t cudaMalloc ( T ** devPtr, size_t size )
{
    return 1;
}

static inline cudaError_t cudaHostAlloc ( void ** pHost, size_t size, unsigned int flags )
{
    return 1;
}

#define cudaHostAllocMapped 0
#define cudaHostAllocPortable 0

static inline cudaError_t cudaFree ( void * devPtr )
{
    return 1;
}

static inline cudaError_t cudaMemcpyAsync ( void * dst, const void * src, size_t count, int kind, int stream = 0 )
{
    return 1;
}

#define cudaMemcpyHostToHost 0
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 0
#define cudaMemcpyDeviceToDevice 0
#define cudaMemcpyDefault 0

struct cudaDeviceProp
{
    int major;
    int minor;
};

static inline cudaError_t cudaGetDeviceProperties ( struct cudaDeviceProp *  prop, int device )
{
    if (!prop)
        return 1;
    memset(prop, 0, sizeof(struct cudaDeviceProp));
    return 0;
}

static inline cudaError_t cudaSetDeviceFlags ( unsigned int flags )
{
    return 1;
}

#define cudaDeviceMapHost 0

static inline cudaError_t cudaSetDevice ( int device )
{
    return 1;
}

static inline float rsqrtf(float f)
{
    return 1.0f / sqrtf(f);
}

#endif

#endif
