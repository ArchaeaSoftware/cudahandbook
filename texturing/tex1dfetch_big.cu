/*
 *
 * tex1dfetch_big.cu
 *
 * Microdemo to illustrate how to use tex1dfetch to cover more than
 * the hardware limit of 31 bits worth of address space, using
 * multiple textures.
 *
 * Build with: nvcc -I ../chLib <options> tex1dfetch_fetch_big.cu
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
#include <stdlib.h>

#include <chError.h>

#define CUDA_LG_MAX_TEX1DFETCH_INDEX 27
#define CUDA_MAX_TEX1DFETCH_INDEX (((size_t)1<<CUDA_LG_MAX_TEX1DFETCH_INDEX)-1)

#define CUDA_MAX_TEX1DFETCH_INDEX_SIZE_T ((size_t) 1<<CUDA_LG_MAX_TEX1DFETCH_INDEX)

#define CUDA_LG_MAX_BYTES_INT1 (CUDA_LG_MAX_TEX1DFETCH_INDEX+4)
#define CUDA_LG_MAX_BYTES_INT2 (CUDA_LG_MAX_TEX1DFETCH_INDEX+8)
#define CUDA_LG_MAX_BYTES_INT4 (CUDA_LG_MAX_TEX1DFETCH_INDEX+16)


#define CUDA_MAX_BYTES_INT1 (((size_t) 1 << CUDA_LG_MAX_TEX1DFETCH_INDEX)*sizeof(int1))
#define CUDA_MAX_BYTES_INT2 (((size_t) 1 << CUDA_LG_MAX_TEX1DFETCH_INDEX)*sizeof(int2))
#define CUDA_MAX_BYTES_INT4 (((size_t) 1 << CUDA_LG_MAX_TEX1DFETCH_INDEX)*sizeof(int4))

#define NUM_BLOCKS 2
#define NUM_THREADS 384
#define TOTAL_THREADS NUM_BLOCKS*NUM_THREADS

__device__ int checksumGPU_array[NUM_BLOCKS*NUM_THREADS];

int
checksumGPU()
{
    int sum = 0;
    int host_checksumGPU[NUM_BLOCKS*NUM_THREADS];
    cudaError_t status;
    size_t i;

    cuda(MemcpyFromSymbol(host_checksumGPU, checksumGPU_array, 
        TOTAL_THREADS*sizeof(int)));
    for ( i = 0; i < TOTAL_THREADS; i++ ) {
        sum += host_checksumGPU[i];
    }
    return sum;
Error:
    return 0;
}

__device__ int4
tex4Fetch( cudaTextureObject_t t0, cudaTextureObject_t t1,
           cudaTextureObject_t t2, cudaTextureObject_t t3, size_t index )
{
    int texID = (int) (index>>CUDA_LG_MAX_TEX1DFETCH_INDEX);
    int i = (int) (index & (CUDA_MAX_TEX1DFETCH_INDEX_SIZE_T-1));
    int4 i4;
    
    if ( texID == 0 ) {
        i4 = tex1Dfetch<int4>( t0, i );
    }
    else if ( texID == 1 ) {
        i4 = tex1Dfetch<int4>( t1, i );
    }
    else if ( texID == 2 ) {
        i4 = tex1Dfetch<int4>( t2, i );
    }
    else if ( texID == 3 ) {
        i4 = tex1Dfetch<int4>( t3, i );
    }
    return i4;
}

__global__ void
TexChecksum4( cudaTextureObject_t t0, cudaTextureObject_t t1,
              cudaTextureObject_t t2, cudaTextureObject_t t3, size_t N )
{
    int sum = 0;
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )
    {
        int4 i4 = tex4Fetch( t0, t1, t2, t3, i );
        sum += i4.x + i4.y + i4.z + i4.w;
    }
    checksumGPU_array[blockIdx.x*blockDim.x + threadIdx.x] = sum;
}

__global__ void
TexChecksum2( cudaTextureObject_t tex2, size_t N )
{
    int sum = 0;
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )
    {
        int2 i2 = tex1Dfetch<int2>( tex2, i );
        sum += i2.x + i2.y;
    }
    checksumGPU_array[blockIdx.x*blockDim.x + threadIdx.x] = sum;
}

__global__ void
TexChecksum1( cudaTextureObject_t tex1, size_t N )
{
    int sum = 0;
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
          i < N; 
          i += gridDim.x*blockDim.x )
    {
        int1 i1 = tex1Dfetch<int1>( tex1, i );
        sum += i1.x;
    }
    checksumGPU_array[blockIdx.x*blockDim.x + threadIdx.x] = sum;
}

bool
TexChecksum( int *out, int c, size_t N,
             cudaTextureObject_t tex1, cudaTextureObject_t tex2,
             const cudaTextureObject_t *tex4 )
{
    cudaError_t status;
    bool ret = false;
    int zero[TOTAL_THREADS];

    memset( zero, 0, TOTAL_THREADS*sizeof(int));
    cuda(MemcpyToSymbol( checksumGPU_array, zero, TOTAL_THREADS*sizeof(int) ));
    switch ( c ) {
        case 1:
            TexChecksum1<<<NUM_BLOCKS,NUM_THREADS>>>( tex1, N / sizeof(int) );
            break;
        case 2:
            TexChecksum2<<<NUM_BLOCKS,NUM_THREADS>>>( tex2, N / sizeof(int2) );
            break;
        case 4:
            TexChecksum4<<<NUM_BLOCKS,NUM_THREADS>>>( tex4[0], tex4[1], tex4[2], tex4[3], N / sizeof(int4) );
            break;
        default:
            goto Error;
    }
    if ( cudaSuccess != cudaDeviceSynchronize() )
        goto Error;
    *out = checksumGPU();
    ret = true;
Error:
    return ret;
}

#undef min
#define min(a,b) (((a)<(b)?(a):(b)))

static cudaTextureObject_t
CreateLinearTexObj( const void *devPtr, cudaChannelFormatDesc chanDesc, size_t numBytes )
{
    cudaResourceDesc resDesc = { .resType = cudaResourceTypeLinear };
    cudaTextureDesc texDesc = {};
    cudaTextureObject_t tex = 0;
    resDesc.res.linear.devPtr = (void *) devPtr;
    resDesc.res.linear.desc = chanDesc;
    resDesc.res.linear.sizeInBytes = numBytes;
    cudaCreateTextureObject( &tex, &resDesc, &texDesc, NULL );
    return tex;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    int *hostTex = 0;
    char *deviceTex = 0;
    bool bAllocedHost = false;

    cudaError_t status;
    cudaDeviceProp props;
    int numMb;
    size_t numBytes;
    size_t i;
    int checksumCPU;

    int checksumGPU1;
    int checksumGPU2;
    int checksumGPU4;

    cudaTextureObject_t texObj1 = 0, texObj2 = 0, texObj4[4] = { 0, 0, 0, 0 };

    cuda(SetDeviceFlags(cudaDeviceMapHost));
    cuda(GetDeviceProperties( &props, 0));
    if ( argc != 2 ) {
        printf( "GPU has %d Mb of device memory\n", (int) (props.totalGlobalMem>>20) );
        printf( "Usage: %s <Mb> where Mb is the number of megabytes to test.\n", argv[0] );
        return 0;
    }
    numMb = atoi( argv[1] );
    numBytes = (size_t) numMb << 20;

    // try for device memory first.
    status = cudaMalloc( (void **) &deviceTex, numBytes);
    if ( cudaSuccess == status ) {
        hostTex = (int *) malloc( numBytes );
        if ( ! hostTex ) {
            printf( "malloc() failed\n" );
            goto Error;
        }
    }
    else {
        printf( "Device alloc of %d Mb failed, trying mapped host memory\n", numMb );
        cuda(HostAlloc( (void **) &hostTex, numBytes, cudaHostAllocMapped ) );
        cuda(HostGetDevicePointer( (void **) &deviceTex, hostTex, 0 ) );
        bAllocedHost = true;
    }

    checksumCPU = 0;
    for ( i = 0; i < numBytes/sizeof(float); i++ ) {
        unsigned int u = rand();
        checksumCPU += u;
        hostTex[i] = u;
    }

    printf( "Expected checksum: 0x%x\n", checksumCPU );
    if ( ! bAllocedHost ) {
        cuda(Memcpy( deviceTex, hostTex, numBytes, cudaMemcpyHostToDevice ));
    }
    if ( numBytes <= CUDA_MAX_BYTES_INT1 ) {
        texObj1 = CreateLinearTexObj( deviceTex, cudaCreateChannelDesc<int1>(), numBytes );
        if ( ! TexChecksum( &checksumGPU1, 1, numBytes, texObj1, 0, NULL ) ) {
            printf( "TexCheckSums failed (unsigned int)\n" );
            goto Error;
        }
        printf( "    tex1 checksum: 0x%x\n", checksumGPU1 ); 
    }
    else {
        printf( "    tex1 checksum: (not performed)\n" );
    }
    if ( numBytes <= CUDA_MAX_BYTES_INT2 ) {
        texObj2 = CreateLinearTexObj( deviceTex, cudaCreateChannelDesc<int2>(), numBytes );
        if ( ! TexChecksum( &checksumGPU2, 2, numBytes, 0, texObj2, NULL ) ) {
            printf( "TexCheckSums failed (int2)\n" );
            goto Error;
        }
        printf( "    tex2 checksum: 0x%x\n", checksumGPU2 ); 
    }
    else {
        printf( "    tex2 checksum: (not performed)\n" );
    }

    {
        int iTexture;
        cudaChannelFormatDesc int4Desc = cudaCreateChannelDesc<int4>();
        size_t numInt4s = numBytes / sizeof(int4);
        int numTextures = (numInt4s+CUDA_MAX_TEX1DFETCH_INDEX)>>
            CUDA_LG_MAX_TEX1DFETCH_INDEX;
        size_t Remainder = numBytes & (CUDA_MAX_BYTES_INT4-1);
        if ( ! Remainder ) {
            Remainder = CUDA_MAX_BYTES_INT4;
        }

        size_t texSizes[4];
        char *texBases[4];
        for ( iTexture = 0; iTexture < numTextures; iTexture++ ) {
            texBases[iTexture] = deviceTex+iTexture*CUDA_MAX_BYTES_INT4;
            texSizes[iTexture] = CUDA_MAX_BYTES_INT4;
        }
        texSizes[iTexture-1] = Remainder;
        while ( iTexture < 4 ) {
            texBases[iTexture] = texBases[iTexture-1];
            texSizes[iTexture] = texSizes[iTexture-1];
            iTexture++;
        }
        for ( iTexture = 0; iTexture < 4; iTexture++ ) {
            texObj4[iTexture] = CreateLinearTexObj( texBases[iTexture], int4Desc, texSizes[iTexture] );
        }
        if ( ! TexChecksum( &checksumGPU4, 4, numBytes, 0, 0, texObj4 ) ) {
            printf( "TexCheckSums failed (int4)\n" );
            goto Error;
        }
    }
    printf( "    tex4 checksum: 0x%x\n", checksumGPU4 );
 
    ret = 0;
Error:
    cudaDestroyTextureObject( texObj1 );
    cudaDestroyTextureObject( texObj2 );
    for ( int k = 0; k < 4; k++ ) {
        cudaDestroyTextureObject( texObj4[k] );
    }
    if ( bAllocedHost ) {
        cudaFreeHost( hostTex );
    }
    else {
        free( hostTex );
        if ( deviceTex ) {
            cudaFree( deviceTex );
        }
    }
    return ret;
}
