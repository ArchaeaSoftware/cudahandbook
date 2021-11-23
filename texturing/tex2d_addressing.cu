/*
 *
 * tex2d_addressing.cu
 *
 * Microdemo for 2D texturing addressing modes.
 *
 * Build with: nvcc -I ../chLib <options> tex2d_addressing.cu
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
#include <float.h>
#include <assert.h>

#include <chError.h>

texture<float2, 2, cudaReadModeElementType> tex;

extern "C" __global__ void
TexReadout( 
    float4 *out, 
    size_t Width, 
    size_t Pitch, 
    size_t Height, 
    float2 base, 
    float2 increment )
{
    for ( int row = blockIdx.y*blockDim.y + threadIdx.y;
              row < Height;
              row += blockDim.y*gridDim.y )
    {
        float4 *outrow = (float4 *) ((char *) out+row*Pitch);
        for ( int col = blockIdx.x*blockDim.x + threadIdx.x;
                  col < Width;
                  col += blockDim.x*gridDim.x )
        {
            float4 value;
            float2 texvalue;
            value.x = base.x+(float)col*increment.x;
            value.y = base.y+(float)row*increment.y;

            texvalue = tex2D( tex, value.x, value.y);
            value.z = texvalue.x;
            value.w = texvalue.y;
            outrow[col] = value;
        }
    }
}

template<class T>
void
CreateAndPrintTex( 
    T *initTex, 
    size_t inWidth, size_t inHeight, 
    size_t outWidth, size_t outHeight,
    float2 base, float2 increment, 
    cudaTextureFilterMode filterMode, 
    cudaTextureAddressMode addressModeX,
    cudaTextureAddressMode addressModeY )
{
    T *texContents = 0;
    cudaArray *texArray = 0;
    float4 *outHost = 0, *outDevice = 0;
    cudaError_t status;
    size_t outPitch;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    dim3 blocks, threads;

    // use caller-provided array, if any, to initialize texture
    if ( initTex ) {
        texContents = initTex;
    }
    else {
        // default is to initialize with identity elements
        texContents = (T *) malloc( inWidth*inHeight*sizeof(T) );
        if ( ! texContents )
            goto Error;
        for ( int row = 0; row < inHeight; row++ ) {
            T *rowptr = texContents + row*inWidth;
            for ( int col = 0; col < outHeight; col++ ) {
                T value;
                value.x = (float) col;
                value.y = (float) row;
                rowptr[col] = value;
            }
        }
    }

    cuda(MallocArray( &texArray, 
                                  &channelDesc, 
                                  inWidth, 
                                  inHeight));

    cuda(Memcpy2DToArray( texArray, 0, 0, 
                                      texContents, inWidth*sizeof(T), 
                                      inWidth*sizeof(T), 
                                      inHeight, 
                                      cudaMemcpyHostToDevice));
    cuda(BindTextureToArray(tex, texArray));

    outPitch = outWidth*sizeof(float4);
    outPitch = (outPitch+0x3f)&~0x3f;

    cuda(HostAlloc( (void **) &outHost, outWidth*outPitch, cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &outDevice, outHost, 0 ));

    tex.filterMode = filterMode;
    tex.addressMode[0] = addressModeX;
    tex.addressMode[1] = addressModeY;
    blocks.x = 2;
    blocks.y = 1;
    threads.x = 64; threads.y = 4;
    TexReadout<<<blocks,threads>>>( outDevice, outWidth, outPitch, outHeight, base, increment );
    cuda(DeviceSynchronize());

    for ( int row = 0; row < outHeight; row++ ) {
        float4 *outrow = (float4 *) ((char *) outHost + row*outPitch);
        for ( int col = 0; col < outWidth; col++ ) {
            printf( "(%.1f, %.1f) ", outrow[col].z, outrow[col].w );
        }
        printf( "\n" );
    }
    printf( "\n" );

Error:
    if ( ! initTex ) free( texContents );
    cudaFreeArray( texArray );
    cudaFreeHost( outHost );
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status;

    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(Free(0));

    // go through once each with linear and point filtering
    do {
        tex.normalized = false;
        tex.filterMode = cudaFilterModePoint;
        tex.addressMode[0] = cudaAddressModeClamp;
        tex.addressMode[1] = cudaAddressModeClamp;

        float2 base, increment;
        base.x = 0.0f;//-1.0f;
        base.y = 0.0f;//-1.0f;
        increment.x = 1.0f;
        increment.y = 1.0f;
//        CreateAndPrintTex<float2>( NULL, 8, 8, 8, 8, base, increment, tex.filterMode, tex.addressMode[0], tex.addressMode[1] );

        CreateAndPrintTex<float2>( NULL, 256, 256, 256, 256, base, increment, tex.filterMode, tex.addressMode[0], tex.addressMode[1] );


    } while ( tex.filterMode == cudaFilterModeLinear );

    ret = 0;
Error:
    return ret;
}
