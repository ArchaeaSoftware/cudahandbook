/*
 *
 * tex1dfetch_offset.cu
 *
 * Microdemo for the offset passback parameter when binding
 * a texture to device memory.
 *
 * Build with: nvcc -I ../chLib <options> tex1dfetch_offset.cu
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

#include <chError.h>

#define NUM_FLOATS 4096

texture<float, 1, cudaReadModeElementType> tex;

__global__ void
TexReadout( float *out, size_t offset, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )
    {
        out[i] = tex1Dfetch( tex, i + (int) offset );
    }
}

bool
CheckTex( float *hostOut, const float *in, size_t offset, size_t N )
{
    float *deviceOut;
    cudaError_t status;
    bool ret = false;
    memset( hostOut, 0, N*sizeof(float) );
    cuda(HostGetDevicePointer( (void **) &deviceOut, hostOut, 0 ));
    
    TexReadout<<<2,384>>>( deviceOut, offset>>2, N );
    cuda(DeviceSynchronize());
    for ( int i = 0; i < N; i++ ) {
        if ( in[i] != hostOut[i] ) {
            printf( "Mismatch at index %d\n", i );
            goto Error;
        }
    }
    ret = true;
Error:
    return ret;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    float *deviceTex = 0;
    float fInit[NUM_FLOATS];

    float *foutHost = 0;
    float *foutDevice = 0;
    cudaError_t status;
    cudaDeviceProp props;
    size_t offset;

    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(GetDeviceProperties( &props, 0));
    printf( "Base texture alignment requirement: %d bytes\n", (int) props.textureAlignment );

    for ( int i = 0; i < NUM_FLOATS; i++ ) {
        fInit[i] = (float) i;
    }

    cuda(Malloc( (void **) &deviceTex, 2*NUM_FLOATS*sizeof(float)) );
    cuda(HostAlloc( (void **) &foutHost, NUM_FLOATS*sizeof(float), cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &foutDevice, foutHost, 0 ));

    for ( offset = 0; offset <= NUM_FLOATS/2; offset += 4 )
    {
        size_t texOffset;
        cuda(Memset(deviceTex, 0xcc, 2*NUM_FLOATS*sizeof(float)));
        cuda(Memcpy(deviceTex+offset, fInit, NUM_FLOATS*sizeof(float), cudaMemcpyHostToDevice));

        cuda(BindTexture( &texOffset, tex, deviceTex+offset, NUM_FLOATS*sizeof(float)) );
        printf( "My offset = %d, texture offset = %d\n", (int) offset, (int) texOffset );

        if ( ! CheckTex( foutHost, fInit, texOffset, NUM_FLOATS ) ) {
            goto Error;
        }
    }
    ret = 0;
Error:
    cudaFreeHost( foutHost );
    cudaFree( deviceTex );
    return ret;
}
