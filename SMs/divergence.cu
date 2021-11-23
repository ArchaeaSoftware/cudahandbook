/*
 *
 * divergence.cu
 *
 * Microdemo to measure performance implications of conditional code.
 *
 * Build with: nvcc [--gpu-architecture sm_xx] [-D USE_FLOAT] [-D USE_IF_STATEMENT] divergence.cu
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
void __device__ sumFloats( float *p, size_t N, size_t n )
{
    float f = base;
    for ( size_t i = 0; i < N; i++ ) {
        *p += f;
        p += n;
    }
}

typedef void(*psumFloats)(float *, size_t, size_t);

__device__ psumFloats rgSumFloats[] = {
    sumFloats< 0>, sumFloats< 1>, sumFloats< 2>, sumFloats< 3>,
    sumFloats< 4>, sumFloats< 5>, sumFloats< 6>, sumFloats< 7>,
    sumFloats< 8>, sumFloats< 9>, sumFloats<10>, sumFloats<11>,
    sumFloats<12>, sumFloats<13>, sumFloats<14>, sumFloats<15>,
    sumFloats<16>, sumFloats<17>, sumFloats<18>, sumFloats<19>,
    sumFloats<20>, sumFloats<21>, sumFloats<22>, sumFloats<23>,
    sumFloats<24>, sumFloats<25>, sumFloats<26>, sumFloats<27>,
    sumFloats<28>, sumFloats<29>, sumFloats<30>, sumFloats<31> };

__global__ void
sumFloats_bywarp( float *p, size_t N )
{
    uint32_t warpid = threadIdx.x>>5;
    N /= blockDim.x*gridDim.x;
    rgSumFloats[warpid]( p+threadIdx.x+blockIdx.x*blockDim.x, N, blockDim.x*gridDim.x );
}

__global__ void
sumFloats_bythread( float *p, size_t N )
{
    
}

int
main()
{
    cudaError_t status;
    size_t N = 1024*1024*1024UL;
    float *p = 0;
    float et;
    cudaEvent_t start = 0, stop = 0;

    cuda(Malloc( (void **) &p, N*sizeof(float)) );
    cuda(Memset( p, 0, N*sizeof(float)) );
    cuda(EventCreate( &start ));
    cuda(EventCreate( &stop ));

    cuda(EventRecord( start ));
    sumFloats_bywarp<<<3072,256>>>( p, N );
    cuda(EventRecord( stop ));
    cuda(DeviceSynchronize());
    cuda(EventElapsedTime( &et, start, stop ));
    
    printf( "%.2f ms = %.2f Gops/s\n", et, (double) N*1000.0/et/1e9 );

    cudaFree( p );
    cudaEventDestroy( stop );
    cudaEventDestroy( start );
    return 0;
Error:
    return 1;
}
