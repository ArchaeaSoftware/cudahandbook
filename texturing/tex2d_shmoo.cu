/*
 *
 * tex2d_shmoo.cu
 *
 * Microbenchmark to ascertain optimal block size for 2D texturing.
 *
 * Build with: nvcc -I ../chLib <options> tex2d_shmoo.cu
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
#include <chUtil.h>

#include <cuda.h>

texture<float, 2, cudaReadModeElementType> tex;

extern "C" __global__ void
TexSums( float *out, size_t Width, size_t Height )
{
    float sum = 0.0f;
    for ( int row = blockIdx.y*blockDim.y + threadIdx.y;
              row < Height;
              row += blockDim.y*gridDim.y )
    {
        for ( int col = blockIdx.x*blockDim.x + threadIdx.x;
                  col < Width;
                  col += blockDim.x*gridDim.x )
        {
            sum += tex2D( tex, (float) col, (float) row );
        }
    }
    if ( out ) {
        out[blockIdx.x*blockDim.x+threadIdx.x] = sum;
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
tex2D_time( float *ms, cudaArray *array, T value, int threadWidth, int threadHeight, int iterations )
{
    CUarray drvArray = (CUarray) array;
    CUDA_ARRAY_DESCRIPTOR desc;
    cudaEvent_t start = 0;
    cudaEvent_t stop = 0;

    cudaError_t status;
    
    cuda(EventCreate(&start));
    cuda(EventCreate(&stop));
    cuda(BindTextureToArray(tex, array));
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
    cuda(EventRecord(start, 0));
    {
        dim3 threads(threadWidth,threadHeight);
        dim3 blocks = dim3(INTDIVIDE_CEILING(desc.Width, threadWidth), 
                           INTDIVIDE_CEILING(desc.Height, threadHeight));

        for ( int i = 0; i < iterations; i++ ) {
            TexSums<<<blocks,threads>>>( NULL, desc.Width, desc.Height );
        }

    }
    cuda(EventRecord(stop, 0));
    cuda(DeviceSynchronize());
    cuda(EventElapsedTime(ms, start, stop));
Error:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return status;
}

template<class T>
void
ShmooTex2D( 
    size_t Width, size_t Height, int iterations
)
{
    cudaArray *texArray = 0;
    cudaError_t status;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaDeviceProp props;
    dim3 blocks, threads;
    double maxBandwidth = 0.0;
    int minThreadWidth, minThreadHeight;

    cuda(GetDeviceProperties( &props, 0 ) );
    cuda(MallocArray(&texArray, &channelDesc, Width, Height, cudaArraySurfaceLoadStore));

    printf( "\tWidth\n\t" );
    for ( int threadWidth = 4; threadWidth <= 64; threadWidth += 2 ) {
        printf( "%d\t", threadWidth );
    }
    printf( "\n" );
    for ( int threadHeight = 4; threadHeight <= 64; threadHeight += 2 ) {
        printf( "%d\t", threadHeight );
        for ( int threadWidth = 4; threadWidth <= 64; threadWidth += 2 ) {
            int totalThreads = threadWidth*threadHeight;
            if ( totalThreads <= props.maxThreadsPerBlock ) {
                double Bandwidth;
                float ms;
                CUDART_CHECK(tex2D_time( &ms, texArray, 3.141592654f, threadWidth, threadHeight, iterations ));

                // compute bandwidth in gigabytes per second (not gibibytes)
                Bandwidth = (double) iterations*Width*Height*sizeof(T) / (ms/1000.0) / 1e9;
                if ( Bandwidth > maxBandwidth ) {
                    maxBandwidth = Bandwidth;
                    minThreadWidth = threadWidth;
                    minThreadHeight = threadHeight;
                }
                printf( "%5.2f\t", ms );
            }
            else {
                printf( "n/a\t" );
            }
        }
        printf( "\n" );
    }
    printf( "Maximum bandwidth of %.2f G/s achieved with %d x %d blocks\n", 
        maxBandwidth, minThreadWidth, minThreadHeight );
Error:
    cudaFreeArray( texArray );
}

int
main( int argc, char *argv[] )
{
    int ret = 1;

    cudaError_t status;

    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(Free(0));
    ShmooTex2D<float>( 4096, 4096, 10 );

    ret = 0;
Error:
    return ret;
}
