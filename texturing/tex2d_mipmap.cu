/*
 *
 * tex2d_mipmap.cu
 *
 * Microdemo for 2D mipmapped textures and level-of-detail (LOD) sampling.
 *
 * Each mip level is filled with a constant equal to its level index, so with
 * linear mipmap filtering a fractional LOD returns that LOD -- making both the
 * level selection and the inter-level (trilinear) blend directly observable.
 *
 * Build with: nvcc -I ../chLib <options> tex2d_mipmap.cu
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
#include <math.h>

#include <chError.h>

//
// Sample the mipmapped texture at a set of explicit LOD values. Mipmapped
// textures use normalized coordinates, so (0.5, 0.5) is the center of the
// image at every level.
//
extern "C" __global__ void
SampleLODs( float *out, cudaTextureObject_t tex, const float *lod, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        out[i] = tex2DLod<float>( tex, 0.5f, 0.5f, lod[i] );
    }
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status_cudart;
    cudaMipmappedArray_t mipArray = 0;
    cudaTextureObject_t tex = 0;
    float *devLOD = 0, *devOut = 0, *hostLevel = 0;

    const unsigned int baseDim = 8;     // 8x8 base image
    const unsigned int numLevels = 4;   // levels 8x8, 4x4, 2x2, 1x1

    const float hostLOD[] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };
    const int nLOD = (int) (sizeof(hostLOD)/sizeof(hostLOD[0]));
    float hostOut[ sizeof(hostLOD)/sizeof(hostLOD[0]) ];

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent( baseDim, baseDim, 0 ); // 2D: depth 0

    cuda(Free(0));

    cuda(MallocMipmappedArray( &mipArray, &channelDesc, extent, numLevels, 0 ));

    // Fill each mip level with a constant equal to its level index. Level 0
    // is the largest, so allocate one staging buffer up front and reuse it.
    hostLevel = (float *) malloc( baseDim*baseDim*sizeof(float) );
    if ( ! hostLevel )
        goto Error_cudart;
    for ( unsigned int level = 0; level < numLevels; level++ ) {
        cudaArray_t levelArray;
        unsigned int dim = baseDim >> level;   // 8, 4, 2, 1
        for ( unsigned int i = 0; i < dim*dim; i++ )
            hostLevel[i] = (float) level;
        cuda(GetMipmappedArrayLevel( &levelArray, mipArray, level ));
        cuda(Memcpy2DToArray( levelArray, 0, 0, hostLevel,
                              dim*sizeof(float), dim*sizeof(float), dim,
                              cudaMemcpyHostToDevice ));
    }

    // A mipmapped texture requires normalized coordinates. Linear filtering
    // within a level plus linear filtering between levels gives trilinear
    // sampling.
    {
        cudaResourceDesc resDesc = { .resType = cudaResourceTypeMipmappedArray };
        cudaTextureDesc  texDesc = {};
        resDesc.res.mipmap.mipmap = mipArray;
        texDesc.normalizedCoords    = 1;
        texDesc.filterMode          = cudaFilterModeLinear;
        texDesc.mipmapFilterMode    = cudaFilterModeLinear;
        texDesc.addressMode[0]      = cudaAddressModeClamp;
        texDesc.addressMode[1]      = cudaAddressModeClamp;
        texDesc.minMipmapLevelClamp = 0.0f;
        texDesc.maxMipmapLevelClamp = (float) (numLevels - 1);
        cuda(CreateTextureObject( &tex, &resDesc, &texDesc, NULL ));
    }

    cuda(Malloc( &devLOD, nLOD*sizeof(float) ));
    cuda(Malloc( &devOut, nLOD*sizeof(float) ));
    cuda(Memcpy( devLOD, hostLOD, nLOD*sizeof(float), cudaMemcpyHostToDevice ));

    SampleLODs<<<1, nLOD>>>( devOut, tex, devLOD, nLOD );
    cuda(DeviceSynchronize());
    cuda(Memcpy( hostOut, devOut, nLOD*sizeof(float), cudaMemcpyDeviceToHost ));

    printf( "%ux%u mipmapped texture, %u levels; "
            "each level filled with its index.\n\n", baseDim, baseDim, numLevels );
    printf( "  %-6s  %-8s\n", "LOD", "sampled" );
    for ( int i = 0; i < nLOD; i++ ) {
        printf( "  %-6.2f  %-8.3f\n", (double) hostLOD[i], (double) hostOut[i] );
        if ( fabsf( hostOut[i] - hostLOD[i] ) > 1e-4f ) {
            fprintf( stderr, "MISMATCH at LOD %.2f: got %.3f\n",
                     (double) hostLOD[i], (double) hostOut[i] );
            goto Error_cudart;
        }
    }
    printf( "\nSampled value tracks the LOD: level selection and the "
            "trilinear inter-level blend both work.\n" );

    ret = 0;
Error_cudart:
    cudaDestroyTextureObject( tex );
    cudaFreeMipmappedArray( mipArray );
    cudaFree( devLOD );
    cudaFree( devOut );
    free( hostLevel );
    return ret;
}
