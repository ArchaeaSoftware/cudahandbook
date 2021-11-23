/*
 *
 * pageableMemcpyHtoDSynchronous.cu
 *
 * Microdemo that illustrates how necessary CPU/GPU concurrency
 * is for a good-performance pageable memcpy.  Identical to
 * pageableMemcpyHtoD.cu except the event synchronize is in a
 * place that breaks concurrency between the CPU and GPU.
 *
 * A pair of pinned staging buffers are allocated, and after the first
 * staging buffer has been filled, the GPU pulls from one while the
 * CPU fills the other.  CUDA events are used for synchronization.
 *
 * This implementation uses the SSE-optimized memcpy of memcpy16.cpp,
 * so for simplicity, it requires host pointers to be 16-byte aligned.
 *
 * Build with: nvcc -I ../chLib <options> pageableMemcpyHtoD16Synchronous.cu memcpy16.cpp
 * Requires: No minimum SM requirement.
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

#include "chError.h"
#include "chTimer.h"

#define STAGING_BUFFER_SIZE 1048576

void *g_hostBuffers[2];
cudaEvent_t g_events[2];

// these are already defined on some platforms - make our
// own definitions that will work.
#undef min
#undef max
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((b)<(a)?(a):(b))

extern bool memcpy16( void *_dst, const void *_src, size_t N );

void
chMemcpyHtoD( void *device, const void *host, size_t N ) 
{
    cudaError_t status;
    char *dst = (char *) device;
    const char *src = (const char *) host;
    int stagingIndex = 0;
    while ( N ) {
        size_t thisCopySize = min( N, STAGING_BUFFER_SIZE );

//        cuda(EventSynchronize( g_events[stagingIndex] ) );
        memcpy16( g_hostBuffers[stagingIndex], src, thisCopySize ); 
        cuda(MemcpyAsync( dst, g_hostBuffers[stagingIndex], thisCopySize, 
            cudaMemcpyHostToDevice, NULL ) );
        cuda(EventRecord( g_events[1-stagingIndex], NULL ) );
        cuda(EventSynchronize( g_events[1-stagingIndex] ) );
        dst += thisCopySize;
        src += thisCopySize;
        N -= thisCopySize;
        stagingIndex = 1 - stagingIndex;
    }
Error:
    return;
}

bool
TestMemcpy( int *dstDevice, int *srcHost, const int *srcOriginal,
            size_t dstOffset, size_t srcOffset, size_t numInts )
{
    chMemcpyHtoD( dstDevice+dstOffset, srcOriginal+srcOffset, numInts*sizeof(int) );
    cudaMemcpy( srcHost, dstDevice+dstOffset, numInts*sizeof(int), cudaMemcpyDeviceToHost );
    for ( size_t i = 0; i < numInts; i++ ) {
        if ( srcHost[i] != srcOriginal[srcOffset+i] ) {
            return false;
        }
    }
    return true;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int *deviceInt = 0;
    int *hostInt = 0;
    const size_t numInts = 32*1048576;
    const int cIterations = 10;
    int *testVector = 0;
    printf( "Pageable memcpy (16-byte aligned)... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    cuda(HostAlloc( &g_hostBuffers[0], STAGING_BUFFER_SIZE, cudaHostAllocDefault ) );
    cuda(HostAlloc( &g_hostBuffers[1], STAGING_BUFFER_SIZE, cudaHostAllocDefault ) );
    cuda(EventCreate( &g_events[0] ) );
    cuda(EventRecord( g_events[0], 0 ) );  // so it is signaled on first synchronize
    cuda(EventCreate( &g_events[1] ) );
    cuda(EventRecord( g_events[1], 0 ) );  // so it is signaled on first synchronize

    cuda(Malloc( &deviceInt, numInts*sizeof(int) ) );
    cuda(HostAlloc( &hostInt, numInts*sizeof(int), 0 ) );

    testVector = (int *) malloc( numInts*sizeof(int) );
    if ( ! testVector ) {
        printf( "malloc() failed\n" );
        return 1;
    }
    for ( size_t i = 0; i < numInts; i++ ) {
        testVector[i] = rand();
    }

    if ( ! TestMemcpy( deviceInt, hostInt, testVector, 0, 0, numInts ) ) {
        goto Error;
    }
    for ( int i = 0; i < cIterations; i++ ) {
        size_t numInts4 = numInts / 4;
        size_t dstOffset = rand() % (numInts4-1);
        size_t srcOffset = rand() % (numInts4-1);
        size_t intsThisIteration = 1 + rand() % (numInts4-max(dstOffset,srcOffset)-1);
        dstOffset *= 4;
        srcOffset *= 4;
        intsThisIteration *= 4;
        if ( ! TestMemcpy( deviceInt, hostInt, testVector, dstOffset, srcOffset, intsThisIteration ) ) {
            TestMemcpy( deviceInt, hostInt, testVector, dstOffset, srcOffset, intsThisIteration );
            goto Error;
        }
    }

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        chMemcpyHtoD( deviceInt, testVector, numInts*sizeof(int) ) ;
    }
    cuda(DeviceSynchronize() );
    chTimerGetTime( &stop );

    {
        double MBytes = cIterations*numInts*sizeof(int) / 1048576.0;
        double MBpers = MBytes / chTimerElapsedTime( &start, &stop );

        printf( "%.2f MB/s\n", MBpers );
    }

    cudaFree( deviceInt );
    cudaFreeHost( hostInt );
    return 0;
Error:
    printf( "Error\n" );
    return 1;
}
