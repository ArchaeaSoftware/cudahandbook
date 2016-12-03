/*
 *
 * surf1Dmemset.cu
 *
 * Microdemo to illustrate 1D memset via surface store.
 *
 * Build with: nvcc --gpu-architecture sm_20 -I ../chLib <options> surf1Dmemset.cu
 * Requires: SM 2.x for surface load/store.
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

#define NUM_VALUES 16

surface<void, 1> surf1D;

template <typename T>
__global__ void
surf1Dmemset_kernel( T value, int offset, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        surf1Dwrite( value, surf1D, (offset+i)*sizeof(T) );
    }
}

template<typename T>
cudaError_t
surf1Dmemset( cudaArray *array, T value, int offset, size_t N )
{
    cudaError_t status;
    cuda(BindSurfaceToArray(surf1D, array));
    surf1Dmemset_kernel<<<2,384>>>( value, offset, N*sizeof(T) );
Error:
    return status;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    float *foutHost = 0;
    cudaError_t status;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *array = 0;

    cudaDeviceProp prop;

    cuda(GetDeviceProperties(&prop, 0));
    if ( prop.major < 2 ) {
        printf( "This application requires SM 2.x (for surface load/store)\n" );
        goto Error;
    }

    cuda(HostAlloc( 
        (void **) &foutHost, 
        NUM_VALUES*sizeof(float), 
        cudaHostAllocMapped));
    cuda(MallocArray( 
        &array, 
        &channelDesc, 
        NUM_VALUES*sizeof(float), 
        1, 
        cudaArraySurfaceLoadStore ) );

    CUDART_CHECK(surf1Dmemset( array, 3.141592654f, 0, NUM_VALUES ));

    cuda(MemcpyFromArray( 
        foutHost, 
        array, 
        0, 
        0, 
        NUM_VALUES*sizeof(float), 
        cudaMemcpyDeviceToHost ));

    printf( "Surface contents (int form):\n" );
    for ( int i = 0; i < NUM_VALUES; i++ ) {
        printf( "%08x ", *(int *) (&foutHost[i]) );
    }
    printf( "\nSurface contents (int form):\n" );
    for ( int i = 0; i < NUM_VALUES; i++ ) {
        printf( "%E ", foutHost[i] );
    }
    printf( "\n" );
    ret = 0;

Error:
    cudaFreeHost( foutHost );
    return ret;
}
