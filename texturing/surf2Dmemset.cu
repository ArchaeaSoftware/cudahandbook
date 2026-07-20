/*
 *
 * surf2Dmemset.cu
 *
 * Microdemo to illustrate 2D memset via surface store.
 *
 * Build with: nvcc --gpu-architecture sm_20 -I ../chLib <options> surf2Dmemset_shmoo.cu -lcuda
 * (Needs driver API for cuArrayGetDescriptor() ).
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
#include <float.h>
#include <assert.h>

#include <chError.h>

#include <cuda.h>

extern "C" __global__ void
TexReadout( float4 *out, cudaTextureObject_t tex, size_t Width, size_t Pitch, size_t Height, float2 base, float2 increment )
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
            float texvalue;
            value.x = base.x+(float)col*increment.x;
            value.y = base.y+(float)row*increment.y;

            texvalue = tex2D<float>( tex, value.x, value.y);
            value.z = texvalue;
            value.w = texvalue;
            outrow[col] = value;
        }
    }
}

template<typename T>
__global__ void
surf2Dmemset_kernel( cudaSurfaceObject_t surf2D, T value,
                     int xOffset, int yOffset, 
                     int Width, int Height )
{
    for ( int row = blockIdx.y*blockDim.y + threadIdx.y; 
                    row < Height; 
                    row += blockDim.y*gridDim.y ) 
    {
        for ( int col = blockIdx.x*blockDim.x + threadIdx.x;
                  col < Width;
                  col += blockDim.x*gridDim.x )
        {
            surf2Dwrite( value, 
                         surf2D, 
                         (xOffset+col)*sizeof(T), 
                         yOffset+row );
        }
    }
}

size_t
CUarray_format_size( CUarray_format fmt )
{
    switch ( fmt ) {
        case CU_AD_FORMAT_UNSIGNED_INT8:  return 1;
        case CU_AD_FORMAT_UNSIGNED_INT16: return 2;
        case CU_AD_FORMAT_UNSIGNED_INT32: return 4;
        case CU_AD_FORMAT_SIGNED_INT8:    return 1;
        case CU_AD_FORMAT_SIGNED_INT16:   return 2;
        case CU_AD_FORMAT_SIGNED_INT32:   return 4;
        case CU_AD_FORMAT_HALF:           return 2;
        case CU_AD_FORMAT_FLOAT:          return 4;
    }
    return 0;
}

template<typename T>
cudaError_t
surf2DmemsetArray( cudaArray *array, T value )
{
    CUarray drvArray = (CUarray) array;
    CUDA_ARRAY_DESCRIPTOR desc;

    cudaError_t status;
    cudaSurfaceObject_t surfObj = 0;
    int minGridSize, blockSize;
    cudaResourceDesc resDesc = { .resType = cudaResourceTypeArray };
    resDesc.res.array.array = array;
    cuda(CreateSurfaceObject( &surfObj, &resDesc ));
    if ( CUDA_SUCCESS != cuArrayGetDescriptor( &desc, drvArray ) ) {
        status = cudaErrorInvalidValue;
        goto Error;
    }

    //
    // Fail if invoked on a CUDA array containing elements of
    // different size than T
    //
    if ( sizeof(T) != desc.NumChannels*CUarray_format_size(desc.Format) ) {
        status = cudaErrorInvalidValue;
        goto Error;
    }
    cuda(OccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, surf2Dmemset_kernel<T> ));
    surf2Dmemset_kernel<<<minGridSize, blockSize>>>(
        surfObj,
        value, 
        0, 0, // X and Y offset
        desc.Width, 
        desc.Height );
    status = cudaDeviceSynchronize();
Error:
    cudaDestroySurfaceObject( surfObj );
    return status;
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
    cudaArray *texArray = 0;
    float4 *outHost = 0, *outDevice = 0;
    cudaError_t status;
    cudaTextureObject_t tex = 0;
    size_t outPitch;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    dim3 blocks, threads;

    // use 2D memset implemented with surface write to initialize texture

    cuda(MallocArray(&texArray, &channelDesc, inWidth, inHeight, cudaArraySurfaceLoadStore));

    CUDART_CHECK(surf2DmemsetArray( texArray, 3.141592654f ) );

    outPitch = outWidth*sizeof(float4);
    outPitch = (outPitch+0x3f)&~0x3f;

    cuda(HostAlloc( (void **) &outHost, outWidth*outPitch, cudaHostAllocMapped));
    cuda(HostGetDevicePointer( (void **) &outDevice, outHost, 0 ));

    {
        cudaResourceDesc resDesc = { .resType = cudaResourceTypeArray };
        cudaTextureDesc  texDesc = {};
        resDesc.res.array.array = texArray;
        texDesc.filterMode = filterMode;
        texDesc.addressMode[0] = addressModeX;
        texDesc.addressMode[1] = addressModeY;
        cuda(CreateTextureObject( &tex, &resDesc, &texDesc, NULL ));
    }
    blocks.x = 2;
    blocks.y = 1;
    threads.x = 64; threads.y = 4;
    TexReadout<<<blocks,threads>>>( outDevice, tex, outWidth, outPitch, outHeight, base, increment );
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
    cudaDestroyTextureObject( tex );
    cudaFreeArray( texArray );
    cudaFreeHost( outHost );
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status;
    cudaDeviceProp prop;
    cudaTextureFilterMode filterMode = cudaFilterModePoint;
    cudaTextureAddressMode addressMode = cudaAddressModeClamp;

    cuda(GetDeviceProperties(&prop, 0));
    if ( prop.major < 2 ) {
        printf( "This application requires SM 2.x (for surface load/store)\n" );
        goto Error;
    }
    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(Free(0));

    // go through once each with linear and point filtering
    do {
        float2 base, increment;
        base.x = 0.0f;//-1.0f;
        base.y = 0.0f;//-1.0f;
        increment.x = 1.0f;
        increment.y = 1.0f;
//        CreateAndPrintTex<float2>( NULL, 8, 8, 8, 8, base, increment, filterMode, addressMode, addressMode );
        CreateAndPrintTex<float>( NULL, 8, 8, 8, 8, base, increment, filterMode, addressMode, addressMode );

//        CreateAndPrintTex<float>( NULL, 256, 256, 256, 256, base, increment, filterMode, addressMode, addressMode );


    } while ( filterMode == cudaFilterModeLinear );
    ret = 0;

Error:
    return ret;
}
