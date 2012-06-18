/*
 *
 * pageableMemcpyHtoD16.cu
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  All rights reserved.
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

//        CUDART_CHECK( cudaEventSynchronize( g_events[stagingIndex] ) );
        memcpy( g_hostBuffers[stagingIndex], src, thisCopySize ); 
        CUDART_CHECK( cudaMemcpyAsync( dst, g_hostBuffers[stagingIndex], thisCopySize, 
            cudaMemcpyHostToDevice, NULL ) );
        CUDART_CHECK( cudaEventRecord( g_events[1-stagingIndex], NULL ) );
        CUDART_CHECK( cudaEventSynchronize( g_events[1-stagingIndex] ) );
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
    const size_t numInts = 8*1048576;
    const int cIterations = 10;
    int *testVector = 0;
    printf( "Pageable memcpy (16-byte aligned)... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    CUDART_CHECK( cudaHostAlloc( &g_hostBuffers[0], STAGING_BUFFER_SIZE, cudaHostAllocDefault ) );
    CUDART_CHECK( cudaHostAlloc( &g_hostBuffers[1], STAGING_BUFFER_SIZE, cudaHostAllocDefault ) );
    CUDART_CHECK( cudaEventCreate( &g_events[0] ) );
    CUDART_CHECK( cudaEventRecord( g_events[0], 0 ) );  // so it is signaled on first synchronize
    CUDART_CHECK( cudaEventCreate( &g_events[1] ) );
    CUDART_CHECK( cudaEventRecord( g_events[1], 0 ) );  // so it is signaled on first synchronize

    CUDART_CHECK( cudaMalloc( &deviceInt, numInts*sizeof(int) ) );
    CUDART_CHECK( cudaHostAlloc( &hostInt, numInts*sizeof(int), 0 ) );

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
        size_t dstOffset = ~(size_t)0xf & (rand() % (numInts-1));
        size_t srcOffset = ~(size_t)0xf & (rand() % (numInts-1));
        size_t intsThisIteration = 1 + rand() % (numInts-max(dstOffset,srcOffset)-1);
        if ( ! TestMemcpy( deviceInt, hostInt, testVector, dstOffset, srcOffset, intsThisIteration ) ) {
            TestMemcpy( deviceInt, hostInt, testVector, dstOffset, srcOffset, intsThisIteration );
            goto Error;
        }
    }

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        //chMemcpyHtoD( deviceInt, testVector, numInts*sizeof(int) ) ;
        cudaMemcpy( deviceInt, testVector, numInts*sizeof(int), cudaMemcpyHostToDevice );
    }
    CUDART_CHECK( cudaThreadSynchronize() );
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
