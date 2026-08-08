/*
 *
 * mallocAsyncSpeed.cu
 *
 * Microbenchmark comparing the per-iteration cost of allocate-then-free using
 * the synchronous cudaMalloc/cudaFree against the stream-ordered
 * cudaMallocAsync/cudaFreeAsync. The synchronous pair suballocates and, on a
 * miss, calls into the kernel-mode driver and synchronizes the device; the
 * stream-ordered pair queues the work in a stream and satisfies it from a
 * reusable memory pool, so a per-iteration allocate/free loop -- the shape of a
 * per-timestep scratch allocation -- stops stalling.
 *
 * Build with: nvcc -I ../chLib mallocAsyncSpeed.cu
 * Requires: CUDA 11.2+ and a device that supports memory pools
 *           (CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED).
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

#include "chError.h"
#include <chrono>

//
// Per-iteration cost of cudaMalloc followed by cudaFree. Both calls are
// synchronous, so the loop cannot overlap with the GPU.
//
double
syncAllocFree( size_t N, int cIterations )
{
    cudaError_t status_cudart;
    double ret = 0.0;
    void *p = NULL;
    std::chrono::steady_clock::time_point start, stop;

    cuda( Malloc( &p, N ) );        // warm-up
    cuda( Free( p ) );

    start = std::chrono::steady_clock::now();
    for ( int i = 0; i < cIterations; i++ ) {
        cuda( Malloc( &p, N ) );
        cuda( Free( p ) );
    }
    stop = std::chrono::steady_clock::now();

    ret = 1e6*std::chrono::duration<double>(stop-start).count() / cIterations;
Error_cudart:
    return ret;
}

//
// Per-iteration cost of cudaMallocAsync followed by cudaFreeAsync in a stream.
// The calls are stream-ordered and satisfied from a memory pool, so the freed
// block is available to the next iteration without a driver round trip.
//
double
asyncAllocFree( size_t N, int cIterations )
{
    cudaError_t status_cudart;
    double ret = 0.0;
    void *p = NULL;
    cudaStream_t stream = 0;
    std::chrono::steady_clock::time_point start, stop;

    cuda( StreamCreate( &stream ) );
    cuda( MallocAsync( &p, N, stream ) );   // warm-up: populate the pool
    cuda( FreeAsync( p, stream ) );
    cuda( StreamSynchronize( stream ) );

    start = std::chrono::steady_clock::now();
    for ( int i = 0; i < cIterations; i++ ) {
        cuda( MallocAsync( &p, N, stream ) );
        cuda( FreeAsync( p, stream ) );
    }
    cuda( StreamSynchronize( stream ) );
    stop = std::chrono::steady_clock::now();

    ret = 1e6*std::chrono::duration<double>(stop-start).count() / cIterations;
Error_cudart:
    if ( stream ) cudaStreamDestroy( stream );
    return ret;
}

int
main( int argc, char *argv[] )
{
    const int cIterations = 10000;
    const size_t sizes[] = { 4096, (size_t) 1<<20, (size_t) 64<<20 };
    const char *labels[] = { "4 KB", "1 MB", "64 MB" };

    cudaFree( 0 );      // establish a context before timing
    printf( "Allocate+free cost per iteration (%d iterations):\n\n", cIterations );
    printf( "  %-8s %16s %20s %9s\n", "size", "cudaMalloc (us)", "cudaMallocAsync (us)", "speedup" );
    for ( int j = 0; j < 3; j++ ) {
        double s = syncAllocFree( sizes[j], cIterations );
        double a = asyncAllocFree( sizes[j], cIterations );
        printf( "  %-8s %16.2f %20.3f %8.1fx\n", labels[j], s, a, (a > 0.0) ? s/a : 0.0 );
    }
    return 0;
}
