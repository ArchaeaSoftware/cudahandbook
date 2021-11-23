/*
 *
 * nullDtoHMemcpySync.cu
 *
 * Microbenchmark for throughput of synchronous device->host memcpy.
 *
 * Build with: nvcc -I ../chLib <options> nullDtoHMemcpySync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int *deviceInt = 0;
    int *hostInt = 0;
    int cIterations = 1000;
    printf( "Measuring NULL device->host memcpy's (with sync)... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    cuda(Malloc( &deviceInt, sizeof(int) ) );
    cuda(HostAlloc( &hostInt, sizeof(int), 0 ) );

    do {
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            cuda(Memcpy( hostInt, deviceInt, sizeof(int), 
                cudaMemcpyDeviceToHost ) );
        }
        cuda(DeviceSynchronize() );
        chTimerGetTime( &stop );
        cIterations *= 2;
    } while ( chTimerElapsedTime( &start, &stop ) < 0.5f ) ;
    cIterations /= 2;   // one too many

    {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerMemcpy = microseconds / (float) cIterations;

        printf( "%.2f us (%d iterations)\n", usPerMemcpy, cIterations );
    }

    cudaFree( deviceInt );
    cudaFreeHost( hostInt );
    return 0;
Error:
    printf( "Error performing allocation\n" );
    return 1;
}
