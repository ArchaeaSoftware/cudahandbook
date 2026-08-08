/*
 *
 * scopedAtomicSpeed.cu
 *
 * Microbenchmark for the throughput of scoped atomics (cuda::atomic_ref)
 * at block, device, and system thread scope, under heavy contention.
 *
 * Every block hammers one counter with relaxed fetch_add. At block scope
 * the counter lives in shared memory (its natural home); at device and
 * system scope it lives in global memory, one counter per block, so the
 * contention domain -- the set of threads sharing a counter -- is the
 * same in all three runs and only the scope and memory space differ.
 * Comparing block against device isolates the shared-memory/on-SM path
 * from the global/device-coherent path; comparing device against system
 * isolates the scope alone, on identical global memory. Throughput is
 * reported in billions of atomic operations per second (Gatomic/s).
 *
 * Build with the top-level CMake build (needs a CUDA 12.x libcu++).
 * Requires: SM 6.0 or higher.
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

#include <cstdio>
#include <cstdint>

#include <cuda/atomic>

#include "chError.h"

// The block's counter lives in shared memory; block scope orders the
// atomic within the block and no further.
template<cuda::thread_scope Scope>
__global__ void
sharedScopedAtomics( uint32_t *gCounters, uint32_t iters )
{
    __shared__ uint32_t s;
    if ( threadIdx.x == 0 ) {
        s = 0;
    }
    __syncthreads();

    cuda::atomic_ref<uint32_t, Scope> counter( s );
    for ( uint32_t i = 0; i < iters; i++ ) {
        counter.fetch_add( 1, cuda::memory_order_relaxed );
    }
    __syncthreads();

    if ( threadIdx.x == 0 ) {
        gCounters[blockIdx.x] = s;
    }
}

// The block's counter lives in global memory; device or system scope
// determines how widely the operation must be made coherent.
template<cuda::thread_scope Scope>
__global__ void
globalScopedAtomics( uint32_t *gCounters, uint32_t iters )
{
    cuda::atomic_ref<uint32_t, Scope> counter( gCounters[blockIdx.x] );
    for ( uint32_t i = 0; i < iters; i++ ) {
        counter.fetch_add( 1, cuda::memory_order_relaxed );
    }
}

// Time one configuration over nTrials launches, returning the best
// (highest-throughput) result in Gatomic/s, or -1.0 on a verification or
// CUDA failure. bShared selects the shared-memory kernel.
template<cuda::thread_scope Scope, bool bShared>
double
timeScopedAtomics(
    uint32_t *dCounters, uint32_t *hCounters,
    int nBlocks, int nThreads, uint32_t iters, int nTrials )
{
    cudaError_t status_cudart;
    double ret = -1.0;
    cudaEvent_t evStart = nullptr, evStop = nullptr;
    float msBest = 1e30f;
    const uint32_t expected = (uint32_t) nThreads * iters;

    cuda(EventCreate( &evStart ));
    cuda(EventCreate( &evStop ));

    for ( int t = 0; t < nTrials; t++ ) {
        cuda(Memset( dCounters, 0, nBlocks*sizeof(uint32_t) ));
        cuda(EventRecord( evStart ));
        if ( bShared ) {
            sharedScopedAtomics<Scope><<<nBlocks, nThreads>>>( dCounters, iters );
        }
        else {
            globalScopedAtomics<Scope><<<nBlocks, nThreads>>>( dCounters, iters );
        }
        cuda(EventRecord( evStop ));
        cuda(EventSynchronize( evStop ));

        float ms;
        cuda(EventElapsedTime( &ms, evStart, evStop ));
        if ( ms < msBest ) {
            msBest = ms;
        }
    }

    // Every counter must have reached nThreads*iters, or the atomics
    // dropped updates and the timing is meaningless.
    cuda(Memcpy( hCounters, dCounters, nBlocks*sizeof(uint32_t),
                 cudaMemcpyDeviceToHost ));
    for ( int b = 0; b < nBlocks; b++ ) {
        if ( hCounters[b] != expected ) {
            fprintf( stderr, "  verification failed: counter %d = %u, "
                     "expected %u\n", b, hCounters[b], expected );
            goto Error_cudart;
        }
    }

    {
        uint64_t ops = (uint64_t) nBlocks * nThreads * iters;
        ret = (double) ops / ( (double) msBest * 1.0e6 );  // Gatomic/s
    }

Error_cudart:
    if ( evStart ) cudaEventDestroy( evStart );
    if ( evStop  ) cudaEventDestroy( evStop );
    return ret;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status_cudart;
    int ret = 1;
    uint32_t *dCounters = nullptr, *hCounters = nullptr;
    cudaDeviceProp prop;
    int device = 0, nSMs = 0;

    // A block-per-counter grid sized to keep every SM busy, 256 threads
    // contending on each counter, and enough iterations to swamp launch
    // overhead. nThreads*iters must fit in uint32_t (it does: 256*20000).
    const int nThreads = 256;
    const uint32_t iters = 20000;
    const int nTrials = 5;
    int nBlocks;

    cuda(GetDevice( &device ));
    cuda(GetDeviceProperties( &prop, device ));
    nSMs = prop.multiProcessorCount;
    nBlocks = nSMs * 32;

    cuda(Malloc( &dCounters, nBlocks*sizeof(uint32_t) ));
    hCounters = (uint32_t *) malloc( nBlocks*sizeof(uint32_t) );
    if ( ! hCounters ) {
        fprintf( stderr, "host allocation failed\n" );
        goto Error_cudart;
    }

    printf( "%s: %d SMs, %d blocks x %d threads, %u fetch_add/thread, "
            "best of %d\n\n",
            prop.name, nSMs, nBlocks, nThreads, iters, nTrials );

    {
        double gBlock = timeScopedAtomics<cuda::thread_scope_block, true>(
            dCounters, hCounters, nBlocks, nThreads, iters, nTrials );
        double gDevice = timeScopedAtomics<cuda::thread_scope_device, false>(
            dCounters, hCounters, nBlocks, nThreads, iters, nTrials );
        double gSystem = timeScopedAtomics<cuda::thread_scope_system, false>(
            dCounters, hCounters, nBlocks, nThreads, iters, nTrials );

        if ( gBlock < 0.0 || gDevice < 0.0 || gSystem < 0.0 ) {
            goto Error_cudart;
        }

        printf( "  %-28s %8.2f Gatomic/s  (%.2fx)\n",
                "block  scope (shared)",  gBlock,  gBlock / gDevice );
        printf( "  %-28s %8.2f Gatomic/s  (%.2fx)\n",
                "device scope (global)",  gDevice, gDevice / gDevice );
        printf( "  %-28s %8.2f Gatomic/s  (%.2fx)\n",
                "system scope (global)",  gSystem, gSystem / gDevice );
    }

    ret = 0;

Error_cudart:
    if ( hCounters ) free( hCounters );
    if ( dCounters ) cudaFree( dCounters );
    return ret;
}
