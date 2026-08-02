/*
 *
 * devrtMallocSpeed.cu
 *
 * Microbenchmark for the device runtime's in-kernel malloc()/free().
 * Each thread of a kernel allocates (or frees) one buffer, and the
 * per-allocation latency is reported for a range of block sizes and
 * allocation sizes.
 *
 * (The companion mallocSpeed.cu measures the host-side allocation APIs
 * cudaMalloc()/cudaMallocHost(); this program measures the device-side
 * heap that a kernel reaches through malloc() and free().)
 *
 * Build with: nvcc -I ../chLib <options> devrtMallocSpeed.cu
 * Requires: SM 2.0 (Fermi) or later, for in-kernel malloc().
 *
 * Copyright (c) 2013-2026, Archaea Software, LLC.
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

#include "chError.h"

__global__ void
AllocateBuffers( void **out, size_t N )
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    out[i] = malloc( N );
}

__global__ void
FreeBuffers( void **in )
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    free( in[i] );
}

//
// Time one allocate-then-free pass and report the per-thread latency
// (in milliseconds) of each. cBlocks*cThreads buffers of N bytes are
// allocated by AllocateBuffers(), then released by FreeBuffers().
//
cudaError_t
MallocSpeed( double *msPerAlloc, double *msPerFree,
             void **devicePointers, size_t N,
             cudaEvent_t evStart, cudaEvent_t evStop,
             int cBlocks, int cThreads )
{
    float etAlloc, etFree;
    cudaError_t status;

    cuda(EventRecord( evStart ) );
    AllocateBuffers<<<cBlocks,cThreads>>>( devicePointers, N );
    cuda(EventRecord( evStop ) );
    cuda(DeviceSynchronize() );
    cuda(GetLastError() );
    cuda(EventElapsedTime( &etAlloc, evStart, evStop ) );

    cuda(EventRecord( evStart ) );
    FreeBuffers<<<cBlocks,cThreads>>>( devicePointers );
    cuda(EventRecord( evStop ) );
    cuda(DeviceSynchronize() );
    cuda(GetLastError() );
    cuda(EventElapsedTime( &etFree, evStart, evStop ) );

    *msPerAlloc = etAlloc / (double) (cBlocks*cThreads);
    *msPerFree = etFree / (double) (cBlocks*cThreads);

Error:
    return status;
}

static const int g_threadCounts[] = { 32, 64, 128, 256, 512 };
static const int g_cThreadConfigs = 5;
static const int g_cBlocks = 64;   // blocks per launch for the sweeps

//
// Sweep block sizes 32..512 for a fixed allocation size, printing the
// per-alloc and per-free latency (in microseconds) for each.
//
static cudaError_t
Sweep( void **devicePointers, size_t N,
       cudaEvent_t evStart, cudaEvent_t evStop )
{
    cudaError_t status = cudaSuccess;
    double msAlloc, msFree;

    for ( int i = 0; i < g_cThreadConfigs; i++ ) {
        printf( "%d\t\t", g_threadCounts[i] );
    }
    printf( "\n" );
    for ( int i = 0; i < g_cThreadConfigs; i++ ) {
        printf( "alloc\tfree\t" );
    }
    printf( "\n" );
    for ( int i = 0; i < g_cThreadConfigs; i++ ) {
        status = MallocSpeed( &msAlloc, &msFree, devicePointers, N,
                              evStart, evStop, g_cBlocks, g_threadCounts[i] );
        if ( cudaSuccess != status )
            return status;
        printf( "%.2f\t%.2f\t", msAlloc*1e3, msFree*1e3 );
    }
    printf( "\n\n" );
    return cudaSuccess;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    void **devicePointers = 0;
    cudaEvent_t evStart = 0, evStop = 0;
    double msAlloc, msFree;
    // Size the pointer array for the largest configuration below. Declared
    // (and initialized) before the first cuda() so no goto bypasses it.
    size_t maxPointers = (size_t) g_cBlocks * g_threadCounts[g_cThreadConfigs-1];
    if ( maxPointers < 500 ) maxPointers = 500;

    // The device heap that in-kernel malloc() draws from must be sized
    // before the first kernel that calls malloc(). Request a gigabyte.
    cuda(DeviceSetLimit( cudaLimitMallocHeapSize, (size_t) 1<<30 ) );
    cuda(EventCreate( &evStart ) );
    cuda(EventCreate( &evStop ) );
    cuda(Malloc( &devicePointers, maxPointers*sizeof(void *) ) );

    // 1 thread per block, 500 blocks, 1MB allocations.
    status = MallocSpeed( &msAlloc, &msFree, devicePointers, (size_t) 1<<20,
                          evStart, evStop, 500, 1 );
    if ( cudaSuccess != status ) goto Error;
    printf( "Microseconds per alloc/free (1 thread per block):\n" );
    printf( "alloc\tfree\n" );
    printf( "%.2f\t%.2f\t\n\n", msAlloc*1e3, msFree*1e3 );

    // 32-512 threads per block, 12K allocations.
    printf( "Microseconds per alloc/free (32-512 threads per block, 12K allocations):\n" );
    status = Sweep( devicePointers, 12*1024, evStart, evStop );
    if ( cudaSuccess != status ) goto Error;

    // 32-512 threads per block, 64-byte allocations.
    printf( "Microseconds per alloc/free (32-512 threads per block, 64-byte allocations):\n" );
    status = Sweep( devicePointers, 64, evStart, evStop );
    if ( cudaSuccess != status ) goto Error;

Error:
    if ( devicePointers ) cudaFree( devicePointers );
    if ( evStart ) cudaEventDestroy( evStart );
    if ( evStop ) cudaEventDestroy( evStop );
    return (cudaSuccess == status) ? 0 : 1;
}
