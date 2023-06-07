/*
 *
 * nullKernelAsyncGraph.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsyncGraph.cu
 * Requires: CUDA graph availability.
 *
 * Copyright (c) 2023, Archaea Software, LLC.
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
#include <cuda_runtime.h>

#include "chError.h"
#include "chTimer.h"

constexpr int itersPerGraph = 100;

__global__
void
NullKernel()
{
}

cudaError_t
cudaCreateGraphNullKernelLaunches( cudaGraph_t *graph, cudaGraphExec_t *graphInstance, cudaStream_t stream, int cIterations )
{
    cudaError_t status;

    cuda(StreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for ( int i = 0; i < cIterations; ++i ) {
        NullKernel<<<1,1,0,stream>>>();
    }
    cuda(StreamEndCapture(stream, graph));
    cuda(GraphInstantiate(graphInstance, *graph, NULL, NULL, 0));
    return cudaSuccess;
Error:
    return status;
}

double
usPerLaunch( int cIterations )
{
    cudaError_t status;
    double microseconds, ret;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphInstance;
    chTimerTimestamp start, stop;

    cuda(Free(0));
    cuda(StreamCreate( &stream ));
    cuda(CreateGraphNullKernelLaunches( &graph, &graphInstance, stream, itersPerGraph ));

    chTimerGetTime( &start );
    int i;
    for ( i = 0; i < cIterations; i += itersPerGraph ) {
        cuda(GraphLaunch( graphInstance, NULL ));
    }
    cuda(DeviceSynchronize());
    chTimerGetTime( &stop );

    microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    ret = microseconds / (float) i;

Error:
    return (status) ? 0.0 : ret;
}

int
main( int argc, char *argv[] )
{
    const int cIterations = 100000;
    printf( "Measuring asynchronous launch time (launched w graphs)... " ); fflush( stdout );

    printf( "%.2f us\n", usPerLaunch(cIterations) );

    return 0;
}
