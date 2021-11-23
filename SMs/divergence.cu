/*
 *
 * divergence.cu
 *
 * Microdemo to measure performance implications of conditional code.
 *
 * Build with: nvcc [--gpu-architecture sm_xx] divergence.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2021, Archaea Software, LLC.
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
#include <unistd.h>
#include <chError.h>

//
// parameters p and n are expected to account for grid structure
// apply threadIdx and/or blockIdx to p before passing in;
// apply blockDim and/or gridDim to n before passing in.
//
template<int base>
void __device__ sumInts( uint32_t *p, size_t N, size_t n )
{
    for ( size_t i = 0; i < N; i++ ) {
        *p += base;
        p += n;
    }
}

typedef void(*psumInts)(uint32_t *, size_t, size_t);

__device__ psumInts rgSumInts[] = {
    sumInts< 0>, sumInts< 1>, sumInts< 2>, sumInts< 3>,
    sumInts< 4>, sumInts< 5>, sumInts< 6>, sumInts< 7>,
    sumInts< 8>, sumInts< 9>, sumInts<10>, sumInts<11>,
    sumInts<12>, sumInts<13>, sumInts<14>, sumInts<15>,
    sumInts<16>, sumInts<17>, sumInts<18>, sumInts<19>,
    sumInts<20>, sumInts<21>, sumInts<22>, sumInts<23>,
    sumInts<24>, sumInts<25>, sumInts<26>, sumInts<27>,
    sumInts<28>, sumInts<29>, sumInts<30>, sumInts<31> };

template<uint32_t sh>
__global__ void
sumInts_bythread( uint32_t *p, size_t N )
{
    uint32_t warpish_id = threadIdx.x>>sh;
    N /= blockDim.x*gridDim.x;
    rgSumInts[warpish_id&31]( p+threadIdx.x+blockIdx.x*blockDim.x, N, blockDim.x*gridDim.x );
}

template<uint32_t sh>
static double
timeByThreads( uint32_t *p, size_t N )
{
    cudaError_t status;
    float elapsed_time;
    double ret = 0.0;
    cudaEvent_t start = 0, stop = 0;

    cuda(EventCreate( &start ));
    cuda(EventCreate( &stop ));

    cuda(EventRecord( start ));
    sumInts_bythread<sh><<<3072,1024>>>( p, N );
    cuda(EventRecord( stop ));
    cuda(DeviceSynchronize());
    cuda(EventElapsedTime( &elapsed_time, start, stop ));
    ret = N*1000.0/elapsed_time/1e9;
    printf( "%2d threads: %f Gops/s\n", 1<<sh, ret );
Error:
    cudaEventDestroy( stop );
    cudaEventDestroy( start );
    return ret;
}

int
main()
{
    cudaError_t status;
    size_t N = 1024*1024*1024UL;
    uint32_t *p = 0;

    cuda(Malloc( (void **) &p, N*sizeof(uint32_t)) );
    cuda(Memset( p, 0, N*sizeof(uint32_t)) );

    timeByThreads<6>( p, N );
    timeByThreads<5>( p, N );
    timeByThreads<4>( p, N );
    timeByThreads<3>( p, N );
    timeByThreads<2>( p, N );
    timeByThreads<1>( p, N );
    timeByThreads<0>( p, N );

    cudaFree( p );
    return 0;
Error:
    return 1;
}
