/*
 *
 * deviceProps.cu
 *
 * Prints the cudaGetDeviceProperties fields that the benchmark harness
 * records as the submission's hardware fingerprint (bench/run.py runs
 * this program and tees its output into the raw log). The output format
 * is a parser contract with bench/suites.py and bench/validate.py:
 * one "key: value" line per property, append-only -- never rename or
 * reorder existing lines.
 *
 * Bus width, L2 size, and the copy-engine count are only available
 * through cudaGetDeviceProperties (nvidia-smi does not report them);
 * clocks are deliberately NOT printed here because CUDA 13 removed
 * cudaDeviceProp::clockRate and ::memoryClockRate -- the harness reads
 * clocks from nvidia-smi instead.
 *
 * Build with: nvcc -I ../chLib <options> deviceProps.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2026, Archaea Software, LLC.
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

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    cudaDeviceProp prop;
    int device = 0;

    CUDART_CHECK( cudaGetDeviceProperties( &prop, device ) );

    printf( "name: %s\n", prop.name );
    printf( "computeCapability: %d.%d\n", prop.major, prop.minor );
    printf( "multiProcessorCount: %d\n", prop.multiProcessorCount );
    printf( "memoryBusWidth: %d bits\n", prop.memoryBusWidth );
    printf( "l2CacheSize: %d KiB\n", (int) (prop.l2CacheSize / 1024) );
    printf( "asyncEngineCount: %d\n", prop.asyncEngineCount );
    printf( "totalGlobalMem: %d MiB\n",
        (int) (prop.totalGlobalMem / 1048576) );

    return 0;
Error:
    fprintf( stderr, "CUDA error %d (%s)\n", status,
        cudaGetErrorString( status ) );
    return 1;
}
