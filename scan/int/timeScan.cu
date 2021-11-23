/*
 *
 * timeScan.cu
 *
 * Microbenchmark to time scan performance.
 *
 * Build with: nvcc -I ../../chLib <options> timeScan.cu
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

#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <chTimer.h>
#include <chAssert.h>
#include <chError.h>

#include "scanWarp.cuh"
#include "scanBlock.cuh"

#include "scanZeroPad.cuh"

#define min(a,b) ((a)<(b)?(a):(b))

int *g_hostIn, *g_hostOut;


#include "scanFan.cuh"
#include "scanReduceThenScan.cuh"
#include "scanReduceThenScan_0.cuh"
#include "scan2Level.cuh"
#include "ScanThrust.cuh"

void
RandomArray( int *out, size_t N, int modulus )
{
    for ( size_t i = 0; i < N; i++ ) {
        out[i] = rand() % modulus;
    }
}

template<class T>
double
TimeScan( void (*pfnScanGPU)(T *, const T *, size_t, int), 
          size_t N, 
          int numThreads, 
          int cIterations )
{
    chTimerTimestamp start, stop;
    cudaError_t status;

    double ret = 0.0;

    int *inGPU = 0;
    int *outGPU = 0;
    int *inCPU = (int *) malloc( N*sizeof(T) );
    int *outCPU = (int *) malloc( N*sizeof(T) );
    if ( 0==inCPU || 0==outCPU )
        goto Error;
    cuda(Malloc( &inGPU, N*sizeof(T) ) );
    cuda(Malloc( &outGPU, N*sizeof(T) ) );

    RandomArray( inCPU, N, N );
    cuda(Memcpy( inGPU, inCPU, N*sizeof(T), cudaMemcpyHostToDevice ) );
    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        pfnScanGPU( outGPU, inGPU, N, numThreads );
    }
    if ( cudaSuccess != cudaDeviceSynchronize() )
        goto Error;
    chTimerGetTime( &stop );

    // ints per second
    ret = (double) cIterations*N / chTimerElapsedTime( &start, &stop );
    
Error:
    cudaFree( outGPU );
    cudaFree( inGPU );
    free( inCPU );
    free( outCPU );
    return ret;
}

int
main( int argc, char *argv[] )
{
    int maxThreads;

    cudaSetDevice( 0 );
    cudaSetDeviceFlags( cudaDeviceMapHost );

    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, 0 );
        maxThreads = prop.maxThreadsPerBlock;
    }

    printf( "ScanThrust (64M): %.2f Mints/s\n", TimeScan<int>(ScanThrust<int>, 64*1048576, 128, 10)/1048576 );

    printf( "scanFan (64M, 128 threads/block): %.2f Mints/s\n", TimeScan<int>(scanFan<int>, 64*1048576, 128, 10)/1048576 );
    printf( "scanFan (64M, 256 threads/block): %.2f Mints/s\n", TimeScan<int>(scanFan<int>, 64*1048576, 256, 10)/1048576 );
    printf( "scanFan (64M, 512 threads/block): %.2f Mints/s\n", TimeScan<int>(scanFan<int>, 64*1048576, 512, 10)/1048576 );
    if ( maxThreads >= 1024 )
    printf( "scanFan (64M, 1024 threads/block): %.2f Mints/s\n", TimeScan<int>(scanFan<int>, 64*1048576, 1024, 10)/1048576 );

    printf( "scanReduceThenScan (64M, 128 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan<int>, 64*1048576, 128, 10)/1048576 );
    printf( "scanReduceThenScan (64M, 256 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan<int>, 64*1048576, 256, 10)/1048576 );
    printf( "scanReduceThenScan (64M, 512 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan<int>, 64*1048576, 512, 10)/1048576 );
    if ( maxThreads >= 1024 )
    printf( "scanReduceThenScan (64M, 1024 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan<int>, 64*1048576, 1024, 10)/1048576 );

    printf( "scanReduceThenScan_0 (64M, 128 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan_0<int>, 64*1048576, 128, 10)/1048576 );
    printf( "scanReduceThenScan_0 (64M, 256 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan_0<int>, 64*1048576, 256, 10)/1048576 );
    printf( "scanReduceThenScan_0 (64M, 512 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan_0<int>, 64*1048576, 512, 10)/1048576 );
    if ( maxThreads >= 1024 )
    printf( "scanReduceThenScan_0 (64M, 1024 threads/block): %.2f Mints/s\n", TimeScan<int>(scanReduceThenScan_0<int>, 64*1048576, 1024, 10)/1048576 );

    printf( "scan2Level_0 (64M, 128 threads/block): %.2f Mints/s\n", TimeScan<int>(scan2Level<int,true>, 64*1048576, 128, 10)/1048576 );
    printf( "scan2Level_0 (64M, 256 threads/block): %.2f Mints/s\n", TimeScan<int>(scan2Level<int,true>, 64*1048576, 256, 10)/1048576 );
    printf( "scan2Level_0 (64M, 512 threads/block): %.2f Mints/s\n", TimeScan<int>(scan2Level<int,true>, 64*1048576, 512, 10)/1048576 );
    if ( maxThreads >= 1024 )
    printf( "scan2Level_0 (64M, 1024 threads/block): %.2f Mints/s\n", TimeScan<int>(scan2Level<int,true>, 64*1048576, 1024, 10)/1048576 );

}
