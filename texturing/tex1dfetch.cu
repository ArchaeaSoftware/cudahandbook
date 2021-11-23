/*
 *
 * tex1dfetch.cu
 *
 * Microdemo to illustrate how to texture from linear device memory.
 *
 * Build with: nvcc -I ../chLib <options> tex1dfetch.cu
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

#define NUM_FLOATS 16

texture<float, 1, cudaReadModeElementType> tex1;

__global__ void
TexReadout( float *out, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )   
    {
        out[i] = tex1Dfetch( tex1, i );
    }
}

void
PrintTex( float *host, size_t N )
{
    float *device;
    cudaError_t status;
    memset( host, 0, N*sizeof(float) );
    cuda(HostGetDevicePointer( (void **) &device, host, 0 ));
    
    TexReadout<<<2,384>>>( device, N );
    cuda(DeviceSynchronize());
    for ( int i = 0; i < N; i++ ) {
        printf( "%.2f ", host[i] );
    }
    printf( "\n" );
Error:;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    float *p = 0;
    float *finHost;
    float *finDevice;

    float *foutHost;
    float *foutDevice;
    cudaError_t status;
    cudaDeviceProp props;

    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(GetDeviceProperties( &props, 0));
    cuda(Malloc( (void **) &p, NUM_FLOATS*sizeof(float)) );
    cuda(HostAlloc( (void **) &finHost, NUM_FLOATS*sizeof(float), cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &finDevice, finHost, 0 ));

    cuda(HostAlloc( (void **) &foutHost, NUM_FLOATS*sizeof(float), cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &foutDevice, foutHost, 0 ));

    for ( int i = 0; i < NUM_FLOATS; i++ ) {
        finHost[i] = (float) i;
    }

    {
        size_t offset;
        cuda(BindTexture( &offset, tex1, finDevice, NUM_FLOATS*sizeof(float)) );
    }

    PrintTex( foutHost, NUM_FLOATS );

    ret = 0;
Error:
    cudaFree( p );
    cudaFreeHost( finHost );
    cudaFreeHost( foutHost );
    return ret;
}
