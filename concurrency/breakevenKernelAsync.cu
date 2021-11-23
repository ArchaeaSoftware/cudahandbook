/*
 *
 * breakevenKernelAsync.cu
 *
 * Microbenchmark of kernel launch overhead for kernels that
 * do varying amounts of work.
 *
 * Build with: nvcc -I ../chLib <options> breakevenKernelAsync.cu
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

#include "chTimer.h"

__device__ int deviceTime;

__global__
void
WaitKernel( int cycles, bool bWrite )
{
    int start = clock();
    int stop;
    do {
        stop = clock();
    } while ( stop - start < cycles );
    if ( bWrite && threadIdx.x==0 && blockIdx.x==0 ) {
        deviceTime = stop - start;
    }
}

int
main( int argc, char *argv[] )
{
    const int cIterations = 100000;

	// Take a warm-up lap
    chTimerTimestamp start, stop;
    for ( int i = 0; i < cIterations; i++ ) {
        WaitKernel<<<1,1>>>( 0, false );
    }
    cudaDeviceSynchronize();

    printf("Cycles\tus\n" );
    for ( int cycles = 0; cycles < 2500; cycles += 100 ) {
        printf( "%d\t", cycles ); fflush( stdout );
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            WaitKernel<<<1,1>>>( cycles, false );
        }
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) cIterations;

        printf( "%.2f\n", usPerLaunch );
    }


    return 0;
}
