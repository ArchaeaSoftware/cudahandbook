/*
 *
 * concurrencyKernelKernel.cu
 *
 * Microbenchmark to shmoo kernel/kernel concurrency.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_20 <options> concurrencyKernelKernel.cu
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

#include <chAssert.h>
#include <chError.h>
#include <chShmoo.h>
#include <chTimer.h>
#include <chCommandLine.h>

#include "AddKernel.cuh"

#include "TimeSequentialKernelKernel.cuh"
#include "TimeConcurrentKernelKernel.cuh"

int
main( int argc, char *argv[] )
{
    const int numTimes = 256;
    float timesSequential[numTimes];
    float timesSequentialBaseline[numTimes];
    float timesConcurrent[numTimes];

    int unrollFactor = 1;

    int numBlocks;

    chShmooRange streamsRange;
    {
        const int numStreams = 8;
        if ( ! chCommandLineGet( &streamsRange, "Streams", argc, argv ) ) {
            streamsRange.Initialize( numStreams );
        }
    }
    chShmooRange cyclesRange;
    {
        const int minCycles = 8;
        const int maxCycles = 512;
        const int stepCycles = 8;
        cyclesRange.Initialize( minCycles, maxCycles, stepCycles );
        chCommandLineGet( &cyclesRange, "Cycles", argc, argv );
    }
    chCommandLineGet( &unrollFactor, "unrollFactor", argc, argv );

    const size_t numInts = 32*1048576;

    numBlocks = 300;
    printf( "Timing sequential operations (baseline, %d blocks)", numBlocks );
    if ( ! TimeSequentialKernelKernel( timesSequentialBaseline, numInts, cyclesRange, unrollFactor, numBlocks ) ) {
        printf( "TimeSequentialKernel failed\n" );
        return 1;
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

    printf( "Timing sequential operations" );
    if ( ! TimeSequentialKernelKernel( timesSequential, numInts, cyclesRange, unrollFactor, numBlocks ) ) {
        printf( "TimeSequentialKernel failed\n" );
        return 1;
    }

    printf( "\nTiming concurrent operations" );
    if ( ! TimeConcurrentKernelKernel( timesConcurrent, numInts, 
        cyclesRange, streamsRange, unrollFactor, numBlocks ) )
    {
        printf( "TimeConcurrentKernelKernel failed\n" );
        return 1;
    }

    printf( "\n%d integers\n", (int) numInts );
    printf( "Cycles\tSeqBase\tSequential\tConcurrent\tSpeedup\n" );

    int index = 0;
    for ( chShmooIterator cycles(cyclesRange); cycles; cycles++, index++ ) {
        printf( "%d\t%.2f\t%.2f\t%.2f\t%.2f\n", 
            *cycles, timesSequentialBaseline[index],
            timesSequential[index], timesConcurrent[index],
            timesSequential[index] / timesConcurrent[index]  );
    }

    return 0;
}
