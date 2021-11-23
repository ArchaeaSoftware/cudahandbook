/*
 *
 * breakevenHtoDMemcpy.cu
 *
 * Microbenchmark to shmoo CPU overhead of host->device memcpy.
 *
 * Build with: nvcc -I ../chLib <options> breakevenHtoDMemcpy.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

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

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int *deviceInt = 0;
    int *hostInt = 0;
    const int cIterations = 100000;
    const size_t numBytes = 65536;
    const size_t byteIncrement = 4096;

    printf( "D->H memcpy times...\n" ); 
    printf( "Size\tTime (us)\n" );
    fflush( stdout );

    chTimerTimestamp start, stop;

    cuda(Malloc( &deviceInt, numBytes ) );
    cuda(HostAlloc( &hostInt, numBytes, 0 ) );

    for ( size_t byteCount = byteIncrement; 
          byteCount <= numBytes; 
          byteCount += byteIncrement )
    {
        printf( "%d\t", (int) byteCount );
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            cuda(MemcpyAsync( deviceInt, hostInt, byteCount, 
                cudaMemcpyHostToDevice, NULL ) );
        }
        cuda(DeviceSynchronize() );
        chTimerGetTime( &stop );

        {
            double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
            double usPerMemcpy = microseconds / (float) cIterations;
            printf( "%.2f\n", usPerMemcpy );
        }
    }


    cudaFree( deviceInt );
    cudaFreeHost( hostInt );
    return 0;
Error:
    printf( "Error performing allocation\n" );
    return 1;
}
