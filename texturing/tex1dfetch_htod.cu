/*
 *
 * tex1dfetch_htod.cu
 *
 * Microbenchmark to measure performance of texturing from host memory.
 *
 * Build with: nvcc -I ../chLib <options> tex1dfetch_htod.cu
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
#include <assert.h>

#include <chTimer.h>

#include <chError.h>

texture<float, 1> tex;

extern "C" __global__ void
TexReadout( float *out, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )
    {
        out[i] = tex1Dfetch( tex, i );
    }
}

template<class T>
float
MeasureBandwidth( void *out, size_t N, int blocks, int threads )
{
    cudaError_t status;
    chTimerTimestamp start, stop;
    double Bandwidth = 0.0f;

    chTimerGetTime( &start );

    TexReadout<<<2,384>>>( (float *) out, N );
    cuda(DeviceSynchronize());

    chTimerGetTime( &stop );

    Bandwidth = ((double) N*sizeof(T) / chTimerElapsedTime( &start, &stop ))/1048576.0;
Error:
    return (float) Bandwidth;
}

template<class T>
float
ComputeMaximumBandwidth( size_t N )
{
    T *inHost = 0;
    T *inDevice = 0;
    T *outDevice = 0;
    T *outHost = 0;
    cudaError_t status;
    bool ret = false;
    float fMaxBandwidth = 0.0f;
    int cMaxBlocks = 0;
    int cMaxThreads = 0;

    cuda(HostAlloc( (void **) &inHost, N*sizeof(T), cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &inDevice, inHost, 0 ));
    cuda(HostAlloc( (void **) &outHost, N*sizeof(T), 0 ) );

    cuda(Malloc( (void **) &outDevice, N*sizeof(T)));

    for ( int i = 0; i < N; i++ ) {
        inHost[i] = (T) i;
    }

    cuda(BindTexture(NULL, tex, inDevice, cudaCreateChannelDesc<T>(), N*sizeof(T)));

    {
        for ( int cBlocks = 8; cBlocks <= 512; cBlocks += 8 ) {
            for ( int cThreads = 16; cThreads <= 512; cThreads += 16 ) {
                memset( outHost, 0, N*sizeof(float) );
                float bw = MeasureBandwidth<T>( outDevice, N, cBlocks, cThreads );
                if ( bw > fMaxBandwidth ) {
                    fMaxBandwidth = bw;
                    cMaxBlocks = cBlocks;
                    cMaxThreads = cThreads;
                    printf( "New maximum of %.2f M/s reached at %d blocks of %d threads\n",
                        fMaxBandwidth, cMaxBlocks, cMaxThreads );
                }
                cuda(Memcpy( outHost, outDevice, N*sizeof(T), cudaMemcpyDeviceToHost ) );

                for ( int i = 0; i < N; i++ ) {
                    assert( outHost[i] == inHost[i] );
                    if ( outHost[i] != inHost[i] ) {
                        goto Error;
                    }
                }
            }
        }
    }

    ret = true;
Error:
    cudaFreeHost( inHost );
    cudaFree( outDevice );
    cudaFreeHost( outHost );
    return ret;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status;
    float fMaxBW = 0.0f;

    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(Free(0));

    fMaxBW = ComputeMaximumBandwidth<float>(64*1048576);
    printf( "Maximum bandwidth achieved: %.2f\n", fMaxBW );

    ret = 0;
Error:
    return ret;
}
