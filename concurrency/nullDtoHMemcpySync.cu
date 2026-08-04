/*
 *
 * nullDtoHMemcpySync.cu
 *
 * Microbenchmark for throughput of synchronous device->host memcpy.
 *
 * Build with: nvcc -I ../chLib <options> nullDtoHMemcpySync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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
#include <chrono>

int
main( int argc, char *argv[] )
{
    cudaError_t status_cudart;
    int *deviceInt = 0;
    int *hostInt = 0;
    int cIterations = 1000;
    printf( "Measuring NULL device->host memcpy's (with sync)... " ); fflush( stdout );

    std::chrono::steady_clock::time_point start, stop;

    cuda(Malloc( &deviceInt, sizeof(int) ) );
    cuda(HostAlloc( &hostInt, sizeof(int), 0 ) );

    do {
        start = std::chrono::steady_clock::now();
        for ( int i = 0; i < cIterations; i++ ) {
            cuda(Memcpy( hostInt, deviceInt, sizeof(int), 
                cudaMemcpyDeviceToHost ) );
        }
        cuda(DeviceSynchronize() );
        stop = std::chrono::steady_clock::now();
        cIterations *= 2;
    } while ( std::chrono::duration<double>(stop - start).count() < 0.5f ) ;
    cIterations /= 2;   // one too many

    {
        double microseconds = 1e6*std::chrono::duration<double>(stop - start).count();
        double usPerMemcpy = microseconds / (float) cIterations;

        printf( "%.2f us (%d iterations)\n", usPerMemcpy, cIterations );
    }

    cudaFree( deviceInt );
    cudaFreeHost( hostInt );
    return 0;
Error_cudart:
    printf( "Error performing allocation\n" );
    return 1;
}
