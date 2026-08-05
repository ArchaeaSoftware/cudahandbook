/*
 *
 * tex2d_gather.cu
 *
 * Microdemo for texture gather: tex2Dgather() returns the four texels that a
 * bilinear fetch would sample, for a chosen channel.
 *
 * The texture is filled so that texel (col,row) holds row*Width+col, a distinct
 * value per texel. Gathering at a point then returns four decodable values, so
 * both the 2x2 footprint it selected and the order the four texels come back in
 * (.x, .y, .z, .w) are directly observable.
 *
 * Build with: nvcc -I ../chLib <options> tex2d_gather.cu
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

#define TEX_DIM 4    // 4x4 texture

//
// Gather the bilinear 2x2 footprint at each query point. tex2Dgather() returns
// the four texels that tex2D() would blend, as a float4, for channel comp (0
// here, since the texture has one channel).
//
extern "C" __global__ void
Gather( float4 *out, cudaTextureObject_t tex, const float2 *query, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        out[i] = tex2Dgather<float4>( tex, query[i].x, query[i].y, 0 );
    }
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status_cudart;
    cudaArray_t texArray = 0;
    cudaTextureObject_t tex = 0;
    float2 *devQuery = 0;
    float4 *devOut = 0;

    const int W = TEX_DIM, H = TEX_DIM;
    float host[H*W];
    for ( int r = 0; r < H; r++ )
        for ( int c = 0; c < W; c++ )
            host[r*W+c] = (float) (r*W + c);

    // Query points in unnormalized (texel-space) coordinates. Each lands where
    // its 2x2 footprint falls entirely inside the texture, so nothing clamps.
    const float2 hostQuery[] = {
        { 2.0f, 2.0f }, { 3.0f, 2.0f }, { 2.0f, 3.0f }, { 1.0f, 1.0f },
    };
    const int nQ = (int) (sizeof(hostQuery)/sizeof(hostQuery[0]));
    float4 hostOut[ sizeof(hostQuery)/sizeof(hostQuery[0]) ];

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    cuda(Free(0));

    cuda(MallocArray( &texArray, &channelDesc, W, H ));
    cuda(Memcpy2DToArray( texArray, 0, 0, host, W*sizeof(float),
                          W*sizeof(float), H, cudaMemcpyHostToDevice ));

    // Unnormalized coordinates make the texel-space query points read directly;
    // the filter mode is immaterial, since gather returns the footprint texels
    // themselves rather than a filtered blend of them.
    {
        cudaResourceDesc resDesc = { .resType = cudaResourceTypeArray };
        cudaTextureDesc  texDesc = { };
        resDesc.res.array.array  = texArray;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.normalizedCoords = 0;
        cuda(CreateTextureObject( &tex, &resDesc, &texDesc, NULL ));
    }

    cuda(Malloc( &devQuery, nQ*sizeof(float2) ));
    cuda(Malloc( &devOut, nQ*sizeof(float4) ));
    cuda(Memcpy( devQuery, hostQuery, nQ*sizeof(float2), cudaMemcpyHostToDevice ));

    Gather<<<1, nQ>>>( devOut, tex, devQuery, nQ );
    cuda(Memcpy( hostOut, devOut, nQ*sizeof(float4), cudaMemcpyDeviceToHost ));

    printf( "%dx%d texture; texel (col,row) holds row*%d+col.\n\n", W, H, W );
    printf( "tex2Dgather() returns the 2x2 bilinear footprint in (.x,.y,.z,.w):\n\n" );

    for ( int q = 0; q < nQ; q++ ) {
        float x = hostQuery[q].x, y = hostQuery[q].y;
        float comp[4] = { hostOut[q].x, hostOut[q].y, hostOut[q].z, hostOut[q].w };

        // The bilinear footprint is texels (c0,r0)..(c0+1,r0+1), c0 = floor(x-0.5).
        int c0 = (int) floorf( x - 0.5f ), r0 = (int) floorf( y - 0.5f );
        int expect[4] = { r0*W+c0, r0*W+c0+1, (r0+1)*W+c0, (r0+1)*W+c0+1 };

        printf( "  gather (%.1f,%.1f):  ", (double) x, (double) y );
        for ( int k = 0; k < 4; k++ ) {
            int v = (int) (comp[k] + 0.5f);
            printf( ".%c=(%d,%d)=%-2d  ", "xyzw"[k], v % W, v / W, v );
        }

        // Verify the returned set is exactly the footprint (order-independent).
        int ok = 1;
        for ( int e = 0; e < 4; e++ ) {
            int found = 0;
            for ( int k = 0; k < 4; k++ )
                if ( (int) (comp[k] + 0.5f) == expect[e] ) found = 1;
            if ( ! found ) ok = 0;
        }
        printf( "footprint %s\n", ok ? "OK" : "MISMATCH" );
        if ( ! ok ) {
            fprintf( stderr, "gather (%.1f,%.1f): returned set is not the 2x2 footprint\n",
                     (double) x, (double) y );
            goto Error_cudart;
        }
    }
    printf( "\nEach gather returned exactly the four texels of its bilinear footprint.\n" );

    ret = 0;
Error_cudart:
    cudaDestroyTextureObject( tex );
    cudaFreeArray( texArray );
    cudaFree( devQuery );
    cudaFree( devOut );
    return ret;
}
