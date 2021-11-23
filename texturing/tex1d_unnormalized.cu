/*
 *
 * tex1d_unnormalized.cu
 *
 * Microdemo to illustrate how to texture using unnormalized
 * texture coordinates in the range [0..Dim), not [0..1).
 *
 * Build with: nvcc -I ../chLib <options> tex1dfetch_unnormalized.cu
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

texture<float, 1> tex;

extern "C" __global__ void
TexReadout( float2 *out, size_t N, float base, float increment )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )
    {
        float x = base + (float) i * increment;
        out[i].x = x;
        out[i].y = tex1D( tex, x );
    }
}

/*
#ifdef PRINT_HEXADECIMAL_FLOATS
        printf( "(%.2f, 0x%08x)\n", outHost[i].x, *(int *) (&outHost[i].y) );
#else
        printf( "(%.2f, %.2f)\n", outHost[i].x, outHost[i].y );
#endif
*/

template<class T>
void
CreateAndPrintTex( T *initTex, size_t texN, size_t outN, 
    float base, float increment, 
    cudaTextureFilterMode filterMode = cudaFilterModePoint, 
    cudaTextureAddressMode addressMode = cudaAddressModeClamp )
{
    T *texContents = 0;
    cudaArray *texArray = 0;

    float2 *outHost = 0, *outDevice = 0;
    cudaError_t status;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

    // use caller-provided array, if any, to initialize texture
    if ( initTex ) {
        texContents = initTex;
    }
    else {
        // default is to initialize with identity elements
        texContents = (T *) malloc( texN*sizeof(T) );
        if ( ! texContents )
            goto Error;
        for ( int i = 0; i < texN; i++ ) {
            texContents[i] = (T) i;
        }
    }

    cuda(MallocArray(&texArray, &channelDesc, texN));

    cuda(HostAlloc( (void **) &outHost, 
                                outN*sizeof(float2), 
                                cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) 
                                           &outDevice, 
                                           outHost, 0 ));

    cuda(MemcpyToArray( texArray, 
                                    0, 0, 
                                    texContents, 
                                    texN*sizeof(T), 
                                    cudaMemcpyHostToDevice));
    cuda(BindTextureToArray(tex, texArray));

    tex.filterMode = filterMode;
    tex.addressMode[0] = addressMode;
    cuda(HostGetDevicePointer(&outDevice, outHost, 0));
    TexReadout<<<2,384>>>( outDevice, outN, base, increment );
    cuda(DeviceSynchronize());

    for ( int i = 0; i < outN; i++ ) {
        printf( "(%.2f, %.2f)\n", outHost[i].x, outHost[i].y );
    }
    printf( "\n" );

Error:
    if ( ! initTex ) free( texContents );
    if ( texArray ) cudaFreeArray( texArray );
    if ( outHost ) cudaFreeHost( outHost );
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status;
    cuda(SetDeviceFlags(cudaDeviceMapHost));

    //CreateAndPrintTex<float>( NULL, 8, 8, 0.0f, 1.0f, cudaFilterModePoint );
    //CreateAndPrintTex<float>( NULL, 8, 8, 0.0f, 1.0f, cudaFilterModeLinear );
//    CreateAndPrintTex<float>( NULL, 8, 20, 0.9f, 0.01f, cudaFilterModePoint );
    {
        float texData[10];
        for ( int i = 0; i < 10; i++ ) {
            texData[i] = (float) i / 10.0f;
        }
//        CreateAndPrintTex<float>( texData, 10, 10, 0.0f, 1.0f, cudaFilterModePoint );
        CreateAndPrintTex<float>( texData, 10, 10, 1.5f, 0.1f, cudaFilterModeLinear );
    }
    ret = 0;
Error:
    return ret;
}
