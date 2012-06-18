/*
 *
 * concurrencyMemcpyKernelMapped.cu
 *
 * Microbenchmark to shmoo concurrency of asynchronous memcpy
 * and concurrent kernel launches.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_20 <options> concurrencyMemcpyKernelMapped.cu
 * Requires: SM 2.x, if kernel concurrency is desired.
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
#include <stdlib.h>

#include "chAssert.h"
#include "chError.h"
#include "chShmoo.h"
#include "chCommandLine.h"
#include "chTimer.h"

#include "AddKernel.cuh"

#include "TimeConcurrentMemcpyKernel.cuh"
#include "TimeSequentialMemcpyKernelMapped.cuh"
#include "TimeConcurrentKernelMapped.cuh"

int
main( int argc, char *argv[] )
{
    const int numTimes = 256;
    float timesSequential[numTimes];
    float timesConcurrent[numTimes];
    int numBlocks;
    int unrollFactor = 1;
    const size_t numInts = 32*1048576;

    cudaSetDeviceFlags( cudaDeviceMapHost );

    chCommandLineGet( &unrollFactor, "unrollFactor", argc, argv );
    chShmooRange cyclesRange;
    {
        const int minCycles = 8;
        const int maxCycles = 512;
        const int stepCycles = 8;
        cyclesRange.Initialize( minCycles, maxCycles, stepCycles );
        chCommandLineGet( &cyclesRange, "Cycles", argc, argv );
    }

    chShmooRange streamsRange;
    {
        int numStreams = 8;
        if ( ! chCommandLineGet( &streamsRange, "streams", argc, argv ) ) {
            streamsRange.Initialize( numStreams );
        }
    }
        
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties( &props, 0 );
        int multiplier = 16;
        chCommandLineGet( &multiplier, "blocksPerSM", argc, argv );
        numBlocks = props.multiProcessorCount * multiplier;
        printf( "Using %d blocks per SM on GPU with %d SMs = %d blocks\n", multiplier, 
            props.multiProcessorCount, numBlocks );
    }

    printf( "Timing mapped operations" );
    if ( ! TimeSequentialMemcpyKernelMapped( timesSequential, numInts, cyclesRange, numBlocks, unrollFactor ) ) {
        printf( "TimeSequentialMemcpyKernelMapped failed\n" );
        return 1;
    }
    printf( "\nTiming streamed operations" );
    if ( ! TimeConcurrentKernelMapped( timesConcurrent, numInts, cyclesRange, streamsRange, numBlocks, unrollFactor ) ) {
        printf( "TimeConcurrentMemcpyKernel failed\n" );
        return 1;
    }

    printf( "\n%d integers\n", (int) numInts );
    printf( "Cycles\tMapped\tStreamed\tSpeedup\n" );

    int index = 0;
    for ( chShmooIterator cycles(cyclesRange); cycles; cycles++, index++ ) {
        printf( "%d\t%.2f\t%.2f\t%.2f\n", 
            *cycles, timesSequential[index], timesConcurrent[index],
            timesConcurrent[index] / timesSequential[index] );
    }

    return 0;
}
