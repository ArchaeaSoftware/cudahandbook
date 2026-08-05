/*
 *
 * mipmapGenNPP.cu
 *
 * Generate the levels of a mipmap from a pitched device-memory image using
 * NPP's image-resize primitive, then store the pyramid into a mipmapped CUDA
 * array so it can be sampled as a mipmapped texture.
 *
 * CUDA ships no mipmap-generation utility (unlike glGenerateMipmap), so the
 * levels must be produced by the application. NPP's nppiResize_32f_C1R with an
 * area (supersampling) filter is a convenient, high-quality way to do it: each
 * level is the previous level resized to half its width and height.
 *
 * Build with: nvcc -I ../chLib mipmapGenNPP.cu -lnppig -lnppc
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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
#include <math.h>

#include <npp.h>

#include <chError.h>

// Check an NPP status the way chError.h's cuda() macro checks CUDA calls:
// report and jump to the shared cleanup label on failure.
#define npp( expr ) do { \
        NppStatus status_npp = (expr); \
        if ( NPP_SUCCESS != status_npp ) { \
            fprintf( stderr, "NPP failure (line %d of file %s): %s returned %d\n", \
                     __LINE__, __FILE__, #expr, (int) status_npp ); \
            goto Error_cudart; \
        } \
    } while (0);

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status_cudart;
    cudaMipmappedArray_t mipArray = 0;
    Npp32f *levelBuf[16] = {0};   // one pitched device buffer per level
    int levelStep[16] = {0};
    float *hostImage = 0, *hostLevel = 0;

    const int baseDim = 8;        // 8x8 base image
    const int numLevels = 4;      // levels 8x8, 4x4, 2x2, 1x1
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent( baseDim, baseDim, 0 );

    cuda(Free(0));

    // --- Level 0: a pitched device-memory image, value = x + y. ---
    hostImage = (float *) malloc( baseDim*baseDim*sizeof(float) );
    if ( ! hostImage )
        goto Error_cudart;
    for ( int y = 0; y < baseDim; y++ )
        for ( int x = 0; x < baseDim; x++ )
            hostImage[y*baseDim+x] = (float)(x + y);
    {
        size_t pitch;
        cuda(MallocPitch( (void **) &levelBuf[0], &pitch, baseDim*sizeof(float), baseDim ));
        levelStep[0] = (int) pitch;
        cuda(Memcpy2D( levelBuf[0], pitch, hostImage, baseDim*sizeof(float),
                       baseDim*sizeof(float), baseDim, cudaMemcpyHostToDevice ));
    }

    // --- Generate each coarser level by resizing the finer one to half size. ---
    for ( int level = 1; level < numLevels; level++ ) {
        int srcDim = baseDim >> (level-1);   // 8, 4, 2
        int dstDim = baseDim >> level;       // 4, 2, 1
        size_t pitch;
        cuda(MallocPitch( (void **) &levelBuf[level], &pitch, dstDim*sizeof(float), dstDim ));
        levelStep[level] = (int) pitch;
        {
            NppiSize srcSize = { srcDim, srcDim };
            NppiRect srcROI  = { 0, 0, srcDim, srcDim };
            NppiSize dstSize = { dstDim, dstDim };
            NppiRect dstROI  = { 0, 0, dstDim, dstDim };
            npp( nppiResize_32f_C1R(
                     levelBuf[level-1], levelStep[level-1], srcSize, srcROI,
                     levelBuf[level],   levelStep[level],   dstSize, dstROI,
                     NPPI_INTER_SUPER ) );
        }
    }
    cuda(DeviceSynchronize());

    // --- Store the pyramid into a mipmapped array so it can be sampled with
    //     tex2DLod() (see tex2d_mipmap.cu). ---
    cuda(MallocMipmappedArray( &mipArray, &channelDesc, extent, numLevels, 0 ));
    for ( int level = 0; level < numLevels; level++ ) {
        cudaArray_t levelArray;
        int dim = baseDim >> level;
        cuda(GetMipmappedArrayLevel( &levelArray, mipArray, level ));
        cuda(Memcpy2DToArray( levelArray, 0, 0, levelBuf[level], levelStep[level],
                              dim*sizeof(float), dim, cudaMemcpyDeviceToDevice ));
    }

    // --- Read back and print each level. The area filter preserves the mean,
    //     so the 1x1 apex must equal the average of the base image. ---
    hostLevel = (float *) malloc( baseDim*baseDim*sizeof(float) );
    if ( ! hostLevel )
        goto Error_cudart;
    for ( int level = 0; level < numLevels; level++ ) {
        int dim = baseDim >> level;
        cuda(Memcpy2D( hostLevel, dim*sizeof(float), levelBuf[level], levelStep[level],
                       dim*sizeof(float), dim, cudaMemcpyDeviceToHost ));
        printf( "level %d (%dx%d):\n", level, dim, dim );
        for ( int y = 0; y < dim; y++ ) {
            printf( "  " );
            for ( int x = 0; x < dim; x++ )
                printf( "%6.2f", (double) hostLevel[y*dim+x] );
            printf( "\n" );
        }
    }

    {
        float apex = hostLevel[0];             // last read back was the 1x1 apex
        float mean = (float) (baseDim - 1);    // mean of x+y over 0..baseDim-1
        printf( "\napex = %.3f, base image mean = %.3f -- %s\n",
                (double) apex, (double) mean,
                (fabsf( apex - mean ) < 1e-3f) ? "match (the area filter preserves the mean)"
                                               : "MISMATCH" );
        if ( fabsf( apex - mean ) >= 1e-3f )
            goto Error_cudart;
    }

    ret = 0;
Error_cudart:
    cudaFreeMipmappedArray( mipArray );
    for ( int i = 0; i < numLevels; i++ )
        cudaFree( levelBuf[i] );
    free( hostImage );
    free( hostLevel );
    return ret;
}
