/*
 *
 * tex1d_addressing.cu
 *
 * Microdemo to illustrate the workings of the texture addressing modes.
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
#include <float.h>
#include <assert.h>

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

float
ClampUnnormalized( float x, float Dim )
{
    if ( x < 0.0f )
        return 0.0f;
    if ( x >= (Dim-1.0f) )
        return Dim-1.0f;
    return x;
}

float
ClampNormalized( float x, float Dim )
{
    if ( x < 0.0f )
        return 0.0f;
    if ( x >= 1.0f )
        return 1.0f - 1.0f/Dim;
    return x;
}

float
Wrap( float x, float Dim )
{
    return x - floorf(x);
}

float
Mirror( float x, float Dim )
{
    int flip = (int) floorf(x) & 1;
    float ret = x - floorf(x);
    if ( ! flip )
        return ret;
    return 1.0f - ret;
}

float
ApplyAddressingMode( float x, float Dim, cudaTextureAddressMode addrMode )
{
    switch ( addrMode ) {
        case cudaAddressModeClamp: return ClampNormalized( x, Dim );
        case cudaAddressModeWrap: return Wrap( x, Dim );
        case cudaAddressModeMirror: return Mirror( x, Dim );
    }
    // should not get here
    return x;
}

float
PseudoReadTexture( float x, const float *base, float Dim, 
    cudaTextureFilterMode filterMode, cudaTextureAddressMode addrMode )
{
    if ( addrMode == cudaAddressModeBorder ) {
        if ( x < 0.0f || x >= 1.0f ) return 0.0f;
    }
    x = ApplyAddressingMode( x, Dim, addrMode );
    if ( cudaFilterModePoint == filterMode ) {
        return base[(int) (x*Dim)];
    }
    x *= Dim;
    float frac = x - (float) (int) x;
    {
        int frac256 = (int) (frac*256.0f+0.5f);
        frac = frac256/256.0f;
    }
    int index = (int) x;
    return (1.0f - frac)*base[index] + frac*base[index+1];
}

static int g_errors;


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

    cuda(MallocArray(&texArray, &channelDesc, texN/**sizeof(T)*/));

    cuda(MemcpyToArray( texArray, 0, 0, texContents, 
                                  texN*sizeof(T), cudaMemcpyHostToDevice));
    cuda(BindTextureToArray(tex, texArray));

    cuda(HostAlloc( (void **) &outHost, outN*sizeof(float2), cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &outDevice, outHost, 0 ));

    tex.normalized = true;
    tex.filterMode = filterMode;
    tex.addressMode[0] = addressMode;
    TexReadout<<<2,384>>>( outDevice, outN, base, increment );
    cuda(DeviceSynchronize());

    for ( int i = 0; i < outN; i++ ) {
        float x = base+(float)i*increment;
        if ( fabsf(x - outHost[i].x) > 1e5f ) {
            _asm int 3
        }
        float emulated = PseudoReadTexture( x, texContents, (float) texN, filterMode, addressMode );
        if ( outHost[i].y != emulated ) {
            (void) PseudoReadTexture( x, texContents, (float) texN, filterMode, addressMode );
g_errors++;
//            _asm int 3
        }
        printf( "(%.2f, %.2f)\n", outHost[i].x, outHost[i].y );
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
        tex.normalized = true;
        tex.filterMode = tex.filterMode==cudaFilterModePoint ? cudaFilterModeLinear : cudaFilterModePoint;
        tex.addressMode[0] = cudaAddressModeClamp;
        CreateAndPrintTex<float>( NULL, 16, 16, 0.0f, 1.0f, tex.filterMode, tex.addressMode[0] );
        CreateAndPrintTex<float>( NULL, 16, 16, -0.5f, 0.15f, tex.filterMode, tex.addressMode[0] );
        CreateAndPrintTex<float>( NULL, 16, 16, -0.5f, 0.16f, tex.filterMode, tex.addressMode[0] );
        CreateAndPrintTex<float>( NULL, 16, 16, -0.5f, 0.17f, tex.filterMode, tex.addressMode[0] );

        tex.addressMode[0] = cudaAddressModeWrap;
        CreateAndPrintTex<float>( NULL, 16, 16, 0.75f, 0.15f, tex.filterMode, tex.addressMode[0] );
        CreateAndPrintTex<float>( NULL, 16, 80, -4.0, 0.1f, tex.filterMode, tex.addressMode[0] );

        tex.addressMode[0] = cudaAddressModeMirror;
        CreateAndPrintTex<float>( NULL, 16, 16, 0.75f, 0.15f, tex.filterMode, tex.addressMode[0] );

    } while ( tex.filterMode == cudaFilterModeLinear );

#if 0
        {
            float texData[10];
            for ( int i = 0; i < 10; i++ ) {
                texData[i] = (float) i / 10.0f;
            }

            CreateAndPrintTex<float>( texData, 10, 4, 1.5f, 0.25f, cudaFilterModePoint );
            
            CreateAndPrintTex<float>( texData, 10, 4, 1.5f, 0.1f, cudaFilterModeLinear );
        }
        tex.addressMode[0] = cudaAddressModeWrap;
        CreateAndPrintTex<float>( NULL, 16, 16, 0.0f, 1.0f, cudaFilterModePoint );

#if 0

        PrintTex<T>( outHost, outDevice, inHost, texN, base, increment, tex.normalized, tex.addressMode[0] );
#if 1
        PrintTex<T>( outHost, outDevice, inHost, texN, -0.5f, 0.15f, tex.normalized, tex.addressMode[0] );
        PrintTex<T>( outHost, outDevice, inHost, texN, -0.5f, 0.16f, tex.normalized, tex.addressMode[0] );
        PrintTex<T>( outHost, outDevice, inHost, texN, -0.5f, 0.17f, tex.normalized, tex.addressMode[0] );

        tex.addressMode[0] = cudaAddressModeBorder;
        inHost[0] = 0.5f;   // so as not to confuse with border color of 0.0
        PrintTex<T>( outHost, outDevice, inHost, texN, -0.5f, 0.15f, tex.normalized, tex.addressMode[0] );
        PrintTex<T>( outHost, outDevice, inHost, texN, -0.5f, 0.16f, tex.normalized, tex.addressMode[0] );
        PrintTex<T>( outHost, outDevice, inHost, texN, -0.5f, 0.17f, tex.normalized, tex.addressMode[0] );

        if ( tex.normalized ) {

            tex.addressMode[0] = cudaAddressModeWrap;
            PrintTex<T>( foutHost, foutDevice, inHost, texN, 14.5f, 0.15f, tex.normalized, tex.addressMode[0] );

            PrintTex<T>( foutHost, foutDevice, inHost, texN, -0.5f, 0.15f, tex.normalized, tex.addressMode[0] );
            PrintTex<T>( foutHost, foutDevice, inHost, texN, -0.5f, 0.16f, tex.normalized, tex.addressMode[0] );
            PrintTex<T>( foutHost, foutDevice, inHost, texN, -0.5f, 0.17f, tex.normalized, tex.addressMode[0] );

            tex.addressMode[0] = cudaAddressModeMirror;
            inHost[0] = 0.5f;   // so as not to confuse with border color of 0.0
            PrintTex<T>( foutHost, foutDevice, inHost, texN, -0.5f, 0.15f, tex.normalized, tex.addressMode[0] );
            PrintTex<T>( foutHost, foutDevice, inHost, texN, -0.5f, 0.16f, tex.normalized, tex.addressMode[0] );
            PrintTex<T>( foutHost, foutDevice, inHost, texN, -0.5f, 0.17f, tex.normalized, tex.addressMode[0] );
        }
#endif
#endif
        tex.normalized = ! tex.normalized;

    } while ( tex.normalized );
#endif
    ret = 0;

Error:
    return ret;
}
