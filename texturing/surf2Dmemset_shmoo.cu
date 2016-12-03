/*
 *
 * surf2Dmemset_shmoo.cu
 *
 * Microbenchmark to measure performance of 2D memset via surface store.
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
#include <chUtil.h>

#include <cuda.h>

surface<void, 2> surf2D;

template<typename T>
__global__ void
surf2Dmemset_kernel( T value, 
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
            surf2Dwrite( value, surf2D, (xOffset+col)*sizeof(T), yOffset+row );
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
surf2DmemsetArray_time( float *ms, cudaArray *array, T value, int threadWidth, int threadHeight )
{
    CUarray drvArray = (CUarray) array;
    CUDA_ARRAY_DESCRIPTOR desc;
    cudaEvent_t start = 0;
    cudaEvent_t stop = 0;

    cudaError_t status;
    
    cuda(EventCreate(&start));
    cuda(EventCreate(&stop));
    cuda(BindSurfaceToArray(surf2D, array));
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
        
        surf2Dmemset_kernel<<<blocks,threads>>>( value, 
                                                 0, 0, // X and Y offset
                                                 desc.Width, 
                                                 desc.Height );
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
ShmooSurf2Dmemset( 
    size_t Width, size_t Height 
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
    for ( int threadWidth = 4; threadWidth <= 64; threadWidth += 4 ) {
        printf( "%d\t", threadWidth );
    }
    printf( "\n" );
    for ( int threadHeight = 4; threadHeight <= 64; threadHeight += 4 ) {
        printf( "%d\t", threadHeight );
        for ( int threadWidth = 4; threadWidth <= 64; threadWidth += 4 ) {
            int totalThreads = threadWidth*threadHeight;
            if ( totalThreads <= props.maxThreadsPerBlock ) {
                float ms;
                double Bandwidth;
                CUDART_CHECK(surf2DmemsetArray_time( &ms, texArray, 3.141592654f, threadWidth, threadHeight ));

                // compute bandwidth in gigabytes per second (not gibibytes)
                Bandwidth = Width*Height*sizeof(T) / (ms/1000.0) / 1e9;

                if ( Bandwidth > maxBandwidth ) {
                    maxBandwidth = (float) Bandwidth;
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
    cudaDeviceProp prop;

    cuda(GetDeviceProperties(&prop, 0));
    if ( prop.major < 2 ) {
        printf( "This application requires SM 2.x (for surface load/store)\n" );
        goto Error;
    }
    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(Free(0));
    ShmooSurf2Dmemset<float>( 8192, 8192 );
    ret = 0;

Error:
    return ret;
}
