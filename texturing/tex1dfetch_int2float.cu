/*
 *
 * tex1dfetch_int2float.cu
 *
 * Microdemo for the method used by GPU texturing hardware to
 * promote integers to unitized floats.
 *
 * Build with: nvcc -I ../chLib <options> tex1dfetch_int2float.cu
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

#include <chError.h>

texture<signed char, 1, cudaReadModeNormalizedFloat> tex;

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

template<class T> float TexPromoteToFloat( T ) { return 0.0f; }

float
TexPromoteToFloat( signed char c )
{
    if ( c == (signed char) 0x80 ) {
        return -1.0f;
    }
    return (float) c / 127.0f;
}

float
TexPromoteToFloat( short s )
{
    if ( s == (short) 0x8000 ) {
        return -1.0f;
    }
    return (float) s / 32767.0f;
}

float
TexPromoteToFloat( unsigned char uc )
{
    return (float) uc / 255.0f;
}

float
TexPromoteToFloat( unsigned short us )
{
    return (float) us / 65535.0f;
}

template<class T>
void
CheckTexPromoteToFloat( size_t N )
{
    T *inHost, *inDevice;
    float *foutHost, *foutDevice;
    cudaError_t status;

    cuda(HostAlloc( (void **) &inHost, 
                                N*sizeof(T), 
                                cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &inDevice, 
                                           inHost, 
                                           0 ));
    cuda(HostAlloc( (void **) &foutHost, 
                                N*sizeof(float), 
                                cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &foutDevice, 
                                           foutHost, 
                                           0 ));

    for ( int i = 0; i < N; i++ ) {
        inHost[i] = (T) i;
    }
    memset( foutHost, 0, N*sizeof(float) );

    cuda(BindTexture( NULL, 
                      tex, 
                      inDevice, 
                      cudaCreateChannelDesc<T>(), 
                      N*sizeof(T)));
    TexReadout<<<2,384>>>( foutDevice, N );
    cuda(DeviceSynchronize());

    for ( int i = 0; i < N; i++ ) {
        printf( "%.2f ", foutHost[i] );
        assert( foutHost[i] == TexPromoteToFloat( (T) i ) );
    }
    printf( "\n" );
Error:
    cudaFreeHost( inHost );
    cudaFreeHost( foutHost );
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status;

    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(Free(0));
    CheckTexPromoteToFloat<signed char>( 256 );
    CheckTexPromoteToFloat<unsigned char>( 256 );

    CheckTexPromoteToFloat<short>( 65536 );
    CheckTexPromoteToFloat<unsigned short>( 65536 );

    ret = 0;
Error:
    return ret;
}
