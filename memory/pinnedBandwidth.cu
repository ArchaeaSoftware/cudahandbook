/*
 *
 * pinnedBandwidth.cu
 *
 * Measure memory bandwidth between pinned host memory and
 * device memory, both directions. Performs the measurements
 * for all devices in the system.
 *
 * Copyright (c) 2015, Archaea Software, LLC.
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

#include <chError.h>

#include <chCommandLine.h>
#include <chTimer.h>

template< cudaMemcpyKind type >
double
Bandwidth( int iDevice, int cIterations, size_t N )
{
    cudaError_t status;

    double ret = 0.0;
    chTimerTimestamp start, stop;
    void *pHost = 0, *pDevice = 0;

    cuda(SetDevice( iDevice ) );
    cuda(Malloc( &pDevice, N ) );
    cuda(MallocHost( &pHost, N ) );
    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        cuda(MemcpyAsync( pDevice, pHost, N, type, NULL ) );
    }
    cuda(DeviceSynchronize() );
    chTimerGetTime( &stop );
    ret = chTimerBandwidth( &start, &stop, cIterations*N );
Error:
    cudaFree( pDevice );
    cudaFreeHost( pHost );
    return ret;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int cIterations = 100;
    int cMB = 64;
    int deviceCount;

    cuda(GetDeviceCount( &deviceCount ) );
    chCommandLineGet( &cIterations, "iterations", argc, argv );
    chCommandLineGet( &cMB, "MB", argc, argv );
    
    printf( "Transferring %d MB %d times... (all bandwidths in GB/s)\n", cMB, cIterations );
    printf( "Device\tHtoD\tDtoH\n" );
    for ( int iDevice = 0; iDevice < deviceCount; iDevice++ ) {
        printf( "%d\t", iDevice );
        printf( "%.2f\t", Bandwidth<cudaMemcpyHostToDevice>( iDevice, cIterations, cMB*(size_t) 1048576 )/1e9 );
        printf( "%.2f\n", Bandwidth<cudaMemcpyDeviceToHost>( iDevice, cIterations, cMB*(size_t) 1048576 )/1e9 );
    }
    return 0;
Error:
    return 1;
}

