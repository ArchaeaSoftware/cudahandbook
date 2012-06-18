/*
 *
 * concurrencyMemcpyKernel.cu
 *
 * Microbenchmark to shmoo performance improvements due to streaming.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_11 <options> concurrencyMemcpyKernel.cu
 * Requires: SM 1.1 for global atomics.
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

#include "chAssert.h"
#include "chError.h"
#include "chShmoo.h"
#include "chCommandLine.h"
#include "chTimer.h"

#include "AddKernel.cuh"

#include "TimeConcurrentMemcpyKernel.cuh"
#include "TimeSequentialMemcpyKernel.cuh"

int
main( int argc, char *argv[] )
{
    const int numTimes = 256;
    float timesSequentialHtoD[numTimes];
    float timesSequentialKernel[numTimes];
    float timesSequentialDtoH[numTimes];
    float timesSequentialTotal[numTimes];

    float timesConcurrent[numTimes];

    int numBlocks;

    int sizeCmdLine = 32;  // 32M integers by default
    size_t numInts;  // 32M integers by default

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
        const int numStreams = 8;
        if ( ! chCommandLineGet( &streamsRange, "Streams", argc, argv ) ) {
            streamsRange.Initialize( numStreams );
        }
    }
    chCommandLineGet( &sizeCmdLine, "size", argc, argv );
    numInts = (size_t) sizeCmdLine * 1048576;

    {
        cudaDeviceProp props;
        cudaGetDeviceProperties( &props, 0 );
        int multiplier = 16;
        chCommandLineGet( &multiplier, "blocksPerSM", argc, argv );
        numBlocks = props.multiProcessorCount * multiplier;
        printf( "Using %d blocks per SM on GPU with %d SMs = %d blocks\n", multiplier, 
            props.multiProcessorCount, numBlocks );
    }

    printf( "%dM integers\n", numInts>>20 );

    printf( "Timing sequential operations" );
    if ( ! TimeSequentialMemcpyKernel( timesSequentialHtoD, timesSequentialKernel, 
        timesSequentialDtoH, timesSequentialTotal, numInts, cyclesRange, numBlocks ) )
    {
        printf( "TimeSequentialMemcpyKernel failed\n" );
        return 1;
    }
    printf( "\nTiming concurrent operations" );
    if ( ! TimeConcurrentMemcpyKernel( timesConcurrent, numInts, 
                                       cyclesRange, streamsRange,
                                       numBlocks ) )
    {
        printf( "TimeConcurrentMemcpyKernel failed\n" );
        return 1;
    }

    printf( "\nCycles\tHtoD\tKernel\tDtoH\tTotal\tConcurrent\tSpeedup\n" );

    int index = 0;
    for ( chShmooIterator cycles(cyclesRange); cycles; cycles++, index++ ) {
        printf( "%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", 
            *cycles, timesSequentialHtoD[index], timesSequentialKernel[index], 
            timesSequentialDtoH[index], timesSequentialTotal[index], timesConcurrent[index],
            timesSequentialTotal[index] / timesConcurrent[index] );
    }

    return 0;
}
