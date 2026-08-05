/*
 *
 * texcubemap.cu
 *
 * Microdemo for cubemap textures: a six-face cube sampled by direction vector.
 *
 * Each face is filled with a constant equal to its face index, so sampling
 * along the axis that points at a face returns that index -- making the
 * direction-to-face mapping (the largest-magnitude component picks the face)
 * directly observable and checkable.
 *
 * Build with: nvcc -I ../chLib <options> texcubemap.cu
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
// Sample a cubemap along a set of direction vectors. A cubemap is addressed by
// a 3D direction (x,y,z): the largest-magnitude component selects one of the
// six faces, and the other two components, divided by it, give the position on
// that face. texCubemap() applies that convention, the same one OpenGL and
// Direct3D use.
//
extern "C" __global__ void
SampleFaces( float *out, cudaTextureObject_t tex, const float4 *dir, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        float4 d = dir[i];
        out[i] = texCubemap<float>( tex, d.x, d.y, d.z );
    }
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status_cudart;
    cudaArray_t cubeArray = 0;
    cudaTextureObject_t tex = 0;
    float4 *devDir = 0;
    float *devOut = 0, *host = 0;

    const unsigned int faceDim = 4;     // 4x4 faces
    const unsigned int numFaces = 6;

    // One direction per face, each pointing straight at a face center, in the
    // face order CUDA stores them: +X, -X, +Y, -Y, +Z, -Z.
    const float4 hostDir[numFaces] = {
        {  1.0f,  0.0f,  0.0f, 0.0f },   // +X -> face 0
        { -1.0f,  0.0f,  0.0f, 0.0f },   // -X -> face 1
        {  0.0f,  1.0f,  0.0f, 0.0f },   // +Y -> face 2
        {  0.0f, -1.0f,  0.0f, 0.0f },   // -Y -> face 3
        {  0.0f,  0.0f,  1.0f, 0.0f },   // +Z -> face 4
        {  0.0f,  0.0f, -1.0f, 0.0f },   // -Z -> face 5
    };
    float hostOut[numFaces];
    const char *label[numFaces] = { "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    // A cubemap array is a 3D array of six square faces tagged cudaArrayCubemap;
    // the depth extent (6) is the face count.
    cudaExtent extent = make_cudaExtent( faceDim, faceDim, numFaces );

    cuda(Free(0));

    cuda(Malloc3DArray( &cubeArray, &channelDesc, extent, cudaArrayCubemap ));

    // Fill each face with a constant equal to its index and copy all six faces
    // into the cubemap array with a single cudaMemcpy3D. The faces are laid out
    // consecutively in host memory, so they form the depth of the copy.
    host = (float *) malloc( faceDim*faceDim*numFaces*sizeof(float) );
    if ( ! host ) {
        fprintf( stderr, "Out of memory allocating the %ux%ux%u host buffer.\n",
                 faceDim, faceDim, numFaces );
        goto Error_cudart;
    }
    for ( unsigned int face = 0; face < numFaces; face++ )
        for ( unsigned int i = 0; i < faceDim*faceDim; i++ )
            host[face*faceDim*faceDim + i] = (float) face;

    {
        cudaMemcpy3DParms p = { };
        p.srcPtr = make_cudaPitchedPtr( host, faceDim*sizeof(float), faceDim, faceDim );
        p.dstArray = cubeArray;
        p.extent = extent;
        p.kind = cudaMemcpyHostToDevice;
        cuda(Memcpy3D( &p ));
    }

    // A cubemap texture is sampled by direction, so it uses normalized
    // coordinates within each face. Point filtering returns each face's exact
    // value, which is all we need to see which face a direction picked.
    {
        cudaResourceDesc resDesc = { .resType = cudaResourceTypeArray };
        cudaTextureDesc  texDesc = { };
        resDesc.res.array.array = cubeArray;
        texDesc.normalizedCoords = 1;
        texDesc.filterMode       = cudaFilterModePoint;
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        cuda(CreateTextureObject( &tex, &resDesc, &texDesc, NULL ));
    }

    cuda(Malloc( &devDir, numFaces*sizeof(float4) ));
    cuda(Malloc( &devOut, numFaces*sizeof(float) ));
    cuda(Memcpy( devDir, hostDir, numFaces*sizeof(float4), cudaMemcpyHostToDevice ));

    SampleFaces<<<1, numFaces>>>( devOut, tex, devDir, numFaces );
    cuda(DeviceSynchronize());
    cuda(Memcpy( hostOut, devOut, numFaces*sizeof(float), cudaMemcpyDeviceToHost ));

    printf( "%ux%u cubemap, each face filled with its index.\n\n", faceDim, faceDim );
    printf( "  %-9s  %-9s  %-6s\n", "direction", "-> face", "sample" );
    for ( unsigned int f = 0; f < numFaces; f++ ) {
        printf( "  %-9s  %-9u  %-6.1f\n", label[f], f, (double) hostOut[f] );
        if ( fabsf( hostOut[f] - (float) f ) > 1e-4f ) {
            fprintf( stderr, "MISMATCH: direction %s should select face %u, got %.1f\n",
                     label[f], f, (double) hostOut[f] );
            goto Error_cudart;
        }
    }
    printf( "\nEach direction selected its expected face.\n" );

    ret = 0;
Error_cudart:
    cudaDestroyTextureObject( tex );
    cudaFreeArray( cubeArray );
    cudaFree( devDir );
    cudaFree( devOut );
    free( host );
    return ret;
}
