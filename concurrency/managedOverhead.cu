/*
 *
 * managedOverhed.cu
 *
 * Microbenchmark to measure overhead of managed memory.
 * This app illustrates how managed memory coherency appears to be
 * implemented with paging.  Only pages accessed by the host
 * code get copied from device to host.
 *
 * Interestingly, the CUDA driver also does not appear to be
 * doing dirty-bit optimizations, since the null kernel is not
 * actually touching any GPU memory.
 *
 * Build with: nvcc -I ../chLib <options> managedOverhead.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2014, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce thce above copyright 
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

__global__
void
NullKernel()
{
}

const size_t pageSize = 4096;

template<bool bTouch>
double
usPerLaunch( int cIterations, size_t cPages=0 )
{
    cudaError_t status;
    double microseconds, ret;
    chTimerTimestamp start, stop;
    void *p = 0;

    cuda(Free(0) );
    if ( cPages ) {
        cuda(MallocManaged( &p, cPages*pageSize ) );
    }

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        NullKernel<<<1,1>>>();
        cuda(DeviceSynchronize() );
        if ( bTouch && 0 != p ) {
            for ( int iPage = 0; iPage < cPages; iPage++ ) {
                ((volatile unsigned char *) p)[iPage*pageSize] |= 1;
            }
        }
    }
    chTimerGetTime( &stop );

    microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    ret = microseconds / (float) cIterations;
    cudaFree( p );
Error:
    return (status) ? 0.0 : ret;
}

int
main( int argc, char *argv[] )
{
    const int cIterations = 1000;
    printf( "Measuring synchronous launch time...\n" ); fflush( stdout );

    printf( "%8.2f us (0 pages)\n", usPerLaunch<false>(cIterations) );
    for ( size_t cPages = 1; cPages < 8192; cPages *= 2 ) {
        printf( "%8.2f us (%d pages)\n", usPerLaunch<true>(cIterations, cPages), cPages );
    }
    printf( "Without touching memory:\n" );
    for ( size_t cPages = 1; cPages < 8192; cPages *= 2 ) {
        printf( "%8.2f us (%d pages)\n", usPerLaunch<false>(cIterations, cPages), cPages );
    }

    return 0;
}
