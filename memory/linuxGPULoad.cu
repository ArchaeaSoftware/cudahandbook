/*
 *
 * linuxGPULoad.cu
 *
 * Multithreaded Linux application pushes bandwidth to or from one
 * socket using a given GPU.  Periodically reports observed 
 * bandwidth.
 *
 * Runs indefinitely until it detects a 'Q' keystroke.
 * Hitting the space bar will cause it to reset counts.
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
#include <stdlib.h>
#include <time.h>

#ifndef _WIN32
#include <numa.h>
#include <pthread.h>
#include <ctype.h>
#endif

#include <ch_conio.h>

#include <cuda.h>

#include <chTimer.h>
#include <chCommandLine.h>

#include <emmintrin.h>

int g_numNodes;
int g_cIterations = 10;

bool
numNodes( int *p )
{
    if ( numa_available() >= 0 ) {
        *p = numa_max_node() + 1;
        return true;
    }
    return false;
}

double
elapsedTimeCopyToGPU( void *dst, void *src, size_t bytes, int cIterations )
{
    double ret = 0.0f;
    chTimerTimestamp start, end;

    chTimerGetTime( &start );
    {
        for ( int i = 0; i < cIterations; i++ ) {
            if ( cudaSuccess != cudaMemcpyAsync( dst, src, bytes, cudaMemcpyHostToDevice ) )
                goto Error;
        }
    }
    if ( cudaSuccess != cudaDeviceSynchronize() )
        goto Error;
    
    chTimerGetTime( &end );

    ret = chTimerElapsedTime( &start, &end );
Error:
    return ret;
}

double
elapsedTimeCopyFromGPU( void *dst, void *src, size_t bytes, int cIterations )
{
    double ret = 0.0f;
    chTimerTimestamp start, end;

    chTimerGetTime( &start );
    {
        for ( int i = 0; i < cIterations; i++ ) {
            if ( cudaSuccess != cudaMemcpyAsync( dst, src, bytes, cudaMemcpyDeviceToHost ) )
                goto Error;
        }
    }
    if ( cudaSuccess != cudaDeviceSynchronize() )
        goto Error;
    
    chTimerGetTime( &end );

    ret = chTimerElapsedTime( &start, &end );
Error:
    return ret;
}

void *
pageAlignedNumaAlloc( size_t bytes, int node )
{
    void *ret;
    printf( "Allocating on node %d\n", node ); fflush(stdout);
    ret = numa_alloc_onnode( bytes, node );
    return ret;
}

void
pageAlignedNumaFree( void *p, size_t bytes )
{
    numa_free( p, bytes );
}

typedef struct __GPU_BANDWIDTH_PARAMETERS
{
    int device;
    int node;
    bool copyToDevice;
    size_t size;
} GPU_BANDWIDTH_PARAMETERS;

typedef struct __GLOBAL_RUNNING_SUMS {
    pthread_mutex_t mutex;

    double totalTime;
    unsigned long long totalBytes;
    bool bExit;

    size_t size;

} GLOBAL_RUNNING_SUMS;

GLOBAL_RUNNING_SUMS globals;

void *
threadBandwidthToSocket( void *_p )
{
    void *ret = 0;
    GPU_BANDWIDTH_PARAMETERS *p = (GPU_BANDWIDTH_PARAMETERS *) _p;
    void *pDevice = 0;

    void *pHost = pageAlignedNumaAlloc( p->size, p->node );
    if ( ! pHost )
        goto Error;
    if ( cudaSuccess != cudaSetDevice( p->device ) )
        goto Error;
    if ( cudaSuccess != cudaHostRegister( pHost, p->size, 0 ) )
        goto Error;
    if ( cudaSuccess != cudaMalloc( &pDevice, p->size ) )
        goto Error;

    while ( ! globals.bExit ) {
        double et = p->copyToDevice ? 
                elapsedTimeCopyToGPU( pDevice, pHost, p->size, g_cIterations ) : 
                elapsedTimeCopyFromGPU( pHost, pDevice, p->size, g_cIterations );
        if ( 0.0 == et ) {
            printf( "Error during DMA\n" );
            goto Error;
        }
        pthread_mutex_lock( &globals.mutex );
        globals.totalBytes += g_cIterations*p->size;
        globals.totalTime += et;
        pthread_mutex_unlock( &globals.mutex );
    }
    ret = 0;
Error:
    if ( pDevice ) cudaFree( pDevice );
    if ( pHost ) {
        cudaHostUnregister( pHost );
        pageAlignedNumaFree( pHost, p->size );
    }
    pthread_mutex_lock( &globals.mutex );
    globals.bExit = true;
    pthread_mutex_unlock( &globals.mutex );
    return ret;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    int node = 0;
    int device = 0;
    int deviceCount = 0;
    bool bCopyToDevice = false;
    int size = 384; // 384MB buffer size by default

    if ( ! numNodes( &g_numNodes ) ) {
        fprintf( stderr, "Failed to query the number of nodes\n" );
        return 1;
    }
    if ( cudaSuccess != cudaGetDeviceCount( &deviceCount ) ) {
        fprintf( stderr, "Failed to get CUDA device count\n" );
        return 1;
    }
    if ( 0 == deviceCount ) {
        fprintf( stderr, "No CUDA devices available\n" );
        return 1;
    }

    if ( argc == 1 ) {
        printf( "Usage: %s --node <src> --device <device> [--numThreads <count>] [--size size]\n", argv[0] );
        printf( "    --node: specify node to allocate host memory on\n" );
        printf( "    --device: specify GPU (device) to use\n" );
        printf( "    --size: size (in MB) of the buffer\n" );
        printf( "    --iterations <count>: number of memcpy's per timing event (default 10)\n" );
        printf( "    --copyToDevice: if specified, the app performs host->device copies from the given node.\n" );
        printf( "                    The default is to write to the node with device->host copies.\n" );

        printf( "Note: This platform has %d nodes available, numbered 0..%d.\n", g_numNodes, g_numNodes-1 );
        printf( "      This platform has %d devices available, numbered 0..%d.\n", deviceCount, deviceCount-1 );

        printf( "\nThis program runs indefinitely until you quit with the Q key.\n" );
        printf( "The bandwidth reported is a running average. To reset the counters, hit the space key.\n" );
        exit(0);
    }

    chCommandLineGet( &g_cIterations, "iterations", argc, argv );
    chCommandLineGet( &node, "node", argc, argv );
    if ( node < 0 || node >= g_numNodes ) {
        fprintf( stderr, "node must be in the range 0..%d\n", g_numNodes-1 );
        exit(1);
    }

    chCommandLineGet( &device, "device", argc, argv );
    if ( device < 0 || device >= deviceCount ) {
        fprintf( stderr, "device must be in the range 0..%d\n", deviceCount-1 );
        exit(1);
    }
    bCopyToDevice = chCommandLineGetBool( "copyToDevice", argc, argv );
    chCommandLineGet( &size, "size", argc, argv );

    globals.size = size*(size_t) 1048576;

    printf( "%d MB on node %d is being %s by GPU %d\n", size, node, bCopyToDevice?"read":"written", device );

    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init( &attr );
        pthread_mutex_init( &globals.mutex, &attr );
    }

    {
        GPU_BANDWIDTH_PARAMETERS cpuParms;
        pthread_t pThread;
        pthread_attr_t attr;

        cpuParms.node = node;
        cpuParms.device = device;
        cpuParms.size = globals.size;
        cpuParms.copyToDevice = bCopyToDevice;

        pthread_attr_init( &attr );
        if ( 0 != pthread_create( &pThread, &attr, threadBandwidthToSocket, &cpuParms ) ) {
            fprintf( stderr, "Yipes. Thread creation failed\n" );
            exit(1);
        }

    }

    do {

        sleep( 10 );
        pthread_mutex_lock( &globals.mutex );
            printf( "Bandwidth: %.2f GB/s\n", (double) globals.totalBytes /1e9 / globals.totalTime );
            globals.totalBytes = 0;
            globals.totalTime = 0;
            if ( kbhit() ) {
                int ch = getch();
                if ( ch == ' ' ) {
                    printf( "Resetting counts\n" );
                    globals.totalBytes = 0;
                    globals.totalTime = 0;
                }
                if ( toupper(ch) == 'Q' ) {
                    printf( "Quitting\n" );
                    globals.bExit = true;
                }
            }
        pthread_mutex_unlock( &globals.mutex );
    } while ( ! globals.bExit );
    ret = 0;

//Error:
    return ret;
}
