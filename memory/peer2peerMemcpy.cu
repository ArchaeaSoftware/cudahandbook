/*
 *
 * peer2peerMemcpy.cu
 *
 * Sample shows how to use portable pinned memory and inter-
 * GPU synchronization to perform a peer-to-peer memcpy.
 *
 * Build with: nvcc -I ../chLib <options> peer2peerMemcpy.cu
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

#define MAX_DEVICES 32

#define STAGING_BUFFER_SIZE 1048576

void *g_hostBuffers[MAX_DEVICES];

// Indexed as follows: [device][event]
cudaEvent_t g_events[MAX_DEVICES][2];
bool g_bEnabled[MAX_DEVICES][MAX_DEVICES];

// these are already defined on some platforms - make our
// own definitions that will work.
#undef min
#undef max
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((b)<(a)?(a):(b))

cudaError_t
chMemcpyPeerToPeer( 
    void *_dst, int dstDevice, 
    const void *_src, int srcDevice, 
    size_t N ) 
{
    cudaError_t status;
    char *dst = (char *) _dst;
    const char *src = (const char *) _src;
    int stagingIndex = 0;
    while ( N ) {
        size_t thisCopySize = min( N, STAGING_BUFFER_SIZE );

        CUDART_CHECK( cudaSetDevice( srcDevice ) );
        CUDART_CHECK( cudaStreamWaitEvent( NULL, g_events[dstDevice][stagingIndex], 0 ) );
        CUDART_CHECK( cudaMemcpyAsync( g_hostBuffers[stagingIndex], src, thisCopySize, 
            cudaMemcpyDeviceToHost, NULL ) );
        CUDART_CHECK( cudaEventRecord( g_events[srcDevice][stagingIndex] ) );

        CUDART_CHECK( cudaSetDevice( dstDevice ) );
        CUDART_CHECK( cudaStreamWaitEvent( NULL, g_events[srcDevice][stagingIndex], 0 ) );
        CUDART_CHECK( cudaMemcpyAsync( dst, g_hostBuffers[stagingIndex], thisCopySize, 
            cudaMemcpyHostToDevice, NULL ) );
        CUDART_CHECK( cudaEventRecord( g_events[dstDevice][stagingIndex] ) );

        dst += thisCopySize;
        src += thisCopySize;
        N -= thisCopySize;
        stagingIndex = 1 - stagingIndex;
    }
    // Wait until both devices are done
    CUDART_CHECK( cudaSetDevice( srcDevice ) );
    CUDART_CHECK( cudaDeviceSynchronize() );

    CUDART_CHECK( cudaSetDevice( dstDevice ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    
Error:
    return status;
}

bool
TestMemcpy( 
    int *dst, int dstDevice,
    int *src, int srcDevice,
    int *srcHost, const int *srcOriginal,
    size_t dstOffset, size_t srcOffset, 
    size_t numInts )
{
    cudaError_t status;

    memset( srcHost, 0, numInts );
    cudaSetDevice( srcDevice );
    CUDART_CHECK( cudaMemcpy( src+srcOffset, srcOriginal+srcOffset, 
        numInts*sizeof(int), cudaMemcpyHostToDevice ) );
    memset( srcHost, 0, numInts*sizeof(int) );
    chMemcpyPeerToPeer( dst+dstOffset, dstDevice, 
                        src+srcOffset, srcDevice, 
                        numInts*sizeof(int) );
    CUDART_CHECK( cudaMemcpy( srcHost, dst+dstOffset, numInts*sizeof(int), cudaMemcpyDeviceToHost ) );
    for ( size_t i = 0; i < numInts; i++ ) {
        if ( srcHost[i] != srcOriginal[srcOffset+i] ) {
            return false;
        }
    }
    return true;
Error:
    return false;
}

int
main( int argc, char *argv[] )
{
    int deviceCount;

    cudaError_t status;
    int *deviceInt[MAX_DEVICES];
    int *hostInt = 0;
    const size_t numInts = 8*1048576;
    const int cIterations = 10;
    int *testVector = 0;
    printf( "Peer-to-peer memcpy... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    memset( deviceInt, 0, sizeof(deviceInt) );

    CUDART_CHECK( cudaGetDeviceCount( &deviceCount ) );

    if ( deviceCount <= 1 ) {
        printf( "Peer-to-peer demo requires at least 2 devices\n" );
        exit(1);
    }

    printf( "%d devices detected\n", deviceCount );

    for ( int i = 0; i < deviceCount; i++ ) {
        cudaSetDevice( i );

        CUDART_CHECK( cudaEventCreate( &g_events[i][0] ) );
        CUDART_CHECK( cudaEventRecord( g_events[i][0], 0 ) );  // so it is signaled on first synchronize
        CUDART_CHECK( cudaEventCreate( &g_events[i][1] ) );
        CUDART_CHECK( cudaEventRecord( g_events[i][1], 0 ) );  // so it is signaled on first synchronize

        CUDART_CHECK( cudaMalloc( &deviceInt[i], numInts*sizeof(int) ) );
    }
    for ( int i = 0; i < deviceCount; i++ ) {
        cudaSetDevice( i );
        for ( int j = 0; j < deviceCount; j++ ) {
            if ( i != j ) {
                int bEnabled;
                CUDART_CHECK( cudaDeviceCanAccessPeer( &bEnabled, i, j ) );
                g_bEnabled[i][j] = (0 != bEnabled);
                if ( bEnabled ) {
                    CUDART_CHECK( cudaDeviceEnablePeerAccess( j, 0 ) );
                }
            }
        }
    }

    CUDART_CHECK( cudaHostAlloc( &g_hostBuffers[0], STAGING_BUFFER_SIZE, cudaHostAllocPortable ) );
    CUDART_CHECK( cudaHostAlloc( &g_hostBuffers[1], STAGING_BUFFER_SIZE, cudaHostAllocPortable ) );

    CUDART_CHECK( cudaHostAlloc( &hostInt, numInts*sizeof(int), 0 ) );

    testVector = (int *) malloc( numInts*sizeof(int) );
    if ( ! testVector ) {
        printf( "malloc() failed\n" );
        return 1;
    }
    for ( size_t i = 0; i < numInts; i++ ) {
        testVector[i] = rand();
    }

    if ( ! TestMemcpy( deviceInt[0], 0, deviceInt[1], 1, 
                       hostInt, testVector, 0, 0, numInts ) ) {
        goto Error;
    }
    for ( int i = 0; i < cIterations; i++ ) {
        size_t dstOffset = rand() % (numInts-1);
        size_t srcOffset = rand() % (numInts-1);
        size_t intsThisIteration = 1 + rand() % (numInts-max(dstOffset,srcOffset)-1);
        if ( ! TestMemcpy( deviceInt[0], 0, deviceInt[1], 1, hostInt, testVector, dstOffset, srcOffset, intsThisIteration ) ) {
            //TestMemcpy( deviceInt, hostInt, testVector, dstOffset, srcOffset, intsThisIteration );
            goto Error;
        }
    }

    for ( int srcDevice = 0; srcDevice < deviceCount; srcDevice++ ) {
        for ( int dstDevice = 0; dstDevice < deviceCount; dstDevice++ ) {
            if ( srcDevice == dstDevice ) continue;
            printf( "%d <- %d...", dstDevice, srcDevice );
            if ( ! g_bEnabled[srcDevice][dstDevice] ) {
                printf( "Not enabled\n" );
                continue;
            }
            chTimerGetTime( &start );
            for ( int i = 0; i < cIterations; i++ ) {
                cudaMemcpyPeerAsync( deviceInt[dstDevice], dstDevice, deviceInt[srcDevice], srcDevice, numInts*sizeof(int) ) ;
            }
            CUDART_CHECK( cudaDeviceSynchronize() );
            chTimerGetTime( &stop );

            {
                double MBytes = cIterations*numInts*sizeof(int) / 1048576.0;
                double MBpers = MBytes / chTimerElapsedTime( &start, &stop );

                printf( "%.2f MB/s\n", MBpers );
            }
        }
    }

    cudaFree( deviceInt );
    cudaFreeHost( hostInt );
    return 0;
Error:
    printf( "Error\n" );
    return 1;
}
