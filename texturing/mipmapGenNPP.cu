/*
 *
 * mipmapGenNPP.cu
 *
 * Generate the levels of a mipmap from a base image in pitched device memory
 * using NPP's image-resize primitive, and store the pyramid into a mipmapped
 * CUDA array so it can be sampled as a mipmapped texture.
 *
 * CUDA ships no mipmap-generation utility (unlike glGenerateMipmap), so the
 * levels are the application's job. GenerateMipmapsNPP() below is a drop-in
 * function you can copy verbatim: it area-resamples each level straight from
 * the base image with nppiResize_32f_C1R (an area/supersampling filter) into a
 * single scratch buffer that is allocated once, ahead of the loop.
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

//
// Populate a single-channel float mipmapped array from a base image held in
// pitched device memory. Level 0 is the base image; every finer level is
// area-resampled straight from the base to its own dimensions and written into
// the array. Because the source is always the base, it never aliases the
// destination, so a single scratch buffer -- allocated once, ahead of the loop
// -- satisfies NPP's no-in-place requirement. Each level is written with a
// tightly packed pitch of its own width; the copy APIs accept any pitch, so
// cudaMallocPitch's padding is only a performance hint we don't need here.
// NPP errors are surfaced as cudaErrorUnknown; the CUDA calls use the book's
// cuda() macro (Section A.6).
//
// eInterpolation selects the resampling filter, since the right filter for
// mipmap generation is application-dependent: NPPI_INTER_SUPER area-averages,
// which for a 2x reduction is the box filter and preserves the image mean;
// NPPI_INTER_CUBIC or NPPI_INTER_LANCZOS give sharper (ringing-prone) pyramids.
//
static cudaError_t
GenerateMipmapsNPP( cudaMipmappedArray_t mipArray,
                    const Npp32f *baseImage, size_t basePitch,
                    int width, int height, int numLevels,
                    int eInterpolation )
{
    cudaError_t status_cudart;
    cudaArray_t levelArray;
    NppiSize srcSize = { width, height };
    NppiRect srcRect = { 0, 0, width, height };
    Npp32f *scratch = 0;

    // Level 0 is the base image itself.
    cuda(GetMipmappedArrayLevel( &levelArray, mipArray, 0 ));
    cuda(Memcpy2DToArray( levelArray, 0, 0, baseImage, basePitch,
                          width*sizeof(Npp32f), height, cudaMemcpyDeviceToDevice ));

    // One scratch buffer, sized for the largest level we resize into (level 1).
    // Every finer level is smaller, so it reuses the top of this buffer.
    if ( numLevels > 1 ) {
        int w1 = ( width > 1 ) ? width/2 : 1, h1 = ( height > 1 ) ? height/2 : 1;
        cuda(Malloc( (void **) &scratch, w1*(size_t) h1*sizeof(Npp32f) ));
    }

    for ( int level = 1; level < numLevels; level++ ) {
        int dstW = ( width  >> level ) ? ( width  >> level ) : 1;
        int dstH = ( height >> level ) ? ( height >> level ) : 1;
        size_t dstPitch = dstW*sizeof(Npp32f);
        NppiSize dstSize = { dstW, dstH };
        NppiRect dstRect = { 0, 0, dstW, dstH };
        NppStatus statusNPP;

        statusNPP = nppiResize_32f_C1R( baseImage, (int) basePitch, srcSize, srcRect,
                                        scratch, (int) dstPitch, dstSize, dstRect,
                                        eInterpolation );
        if ( NPP_SUCCESS != statusNPP ) {
            fprintf( stderr, "nppiResize failed generating level %d: NPP status %d\n",
                     level, (int) statusNPP );
            status_cudart = cudaErrorUnknown;
            goto Error_cudart;
        }
        cuda(GetMipmappedArrayLevel( &levelArray, mipArray, level ));
        cuda(Memcpy2DToArray( levelArray, 0, 0, scratch, dstPitch,
                              dstW*sizeof(Npp32f), dstH, cudaMemcpyDeviceToDevice ));
    }

    status_cudart = cudaSuccess;
Error_cudart:
    cudaFree( scratch );
    return status_cudart;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status_cudart;
    cudaMipmappedArray_t mipArray = 0;
    Npp32f *devImage = 0;       // base image in pitched device memory
    float *host = 0;            // one host buffer, reused for input and readback
    size_t devPitch = 0;

    const int baseDim = 8;      // 8x8 base image
    const int numLevels = 4;    // levels 8x8, 4x4, 2x2, 1x1
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent( baseDim, baseDim, 0 );

    cuda(Free(0));

    // Build a pitched device-memory base image: value = x + y.
    host = (float *) malloc( baseDim*baseDim*sizeof(float) );
    if ( ! host ) {
        fprintf( stderr, "Out of memory allocating the %dx%d host buffer.\n",
                 baseDim, baseDim );
        goto Error_cudart;
    }
    for ( int y = 0; y < baseDim; y++ )
        for ( int x = 0; x < baseDim; x++ )
            host[y*baseDim+x] = (float)(x + y);
    cuda(MallocPitch( (void **) &devImage, &devPitch, baseDim*sizeof(float), baseDim ));
    cuda(Memcpy2D( devImage, devPitch, host, baseDim*sizeof(float),
                   baseDim*sizeof(float), baseDim, cudaMemcpyHostToDevice ));

    // Allocate the mipmapped array and fill every level. NPPI_INTER_SUPER is
    // the area filter -- the box filter for a 2x reduction.
    cuda(MallocMipmappedArray( &mipArray, &channelDesc, extent, numLevels, 0 ));
    CUDART_CHECK( GenerateMipmapsNPP( mipArray, devImage, devPitch,
                                      baseDim, baseDim, numLevels, NPPI_INTER_SUPER ) );

    // Read each level back (reusing the one host buffer) and print it. The area
    // filter preserves the mean, so the 1x1 apex equals the base image average.
    for ( int level = 0; level < numLevels; level++ ) {
        cudaArray_t levelArray;
        int dim = baseDim >> level;
        cuda(GetMipmappedArrayLevel( &levelArray, mipArray, level ));
        cuda(Memcpy2DFromArray( host, dim*sizeof(float), levelArray, 0, 0,
                                dim*sizeof(float), dim, cudaMemcpyDeviceToHost ));
        printf( "level %d (%dx%d):\n", level, dim, dim );
        for ( int y = 0; y < dim; y++ ) {
            printf( "  " );
            for ( int x = 0; x < dim; x++ )
                printf( "%6.2f", (double) host[y*dim+x] );
            printf( "\n" );
        }
    }

    {
        float apex = host[0];               // last read back was the 1x1 apex
        float mean = (float) (baseDim - 1); // mean of x+y over 0..baseDim-1
        printf( "\napex = %.3f, base image mean = %.3f\n", (double) apex, (double) mean );
        if ( fabsf( apex - mean ) >= 1e-3f ) {
            fprintf( stderr, "Verification failed: an area filter preserves the mean, "
                     "so the 1x1 apex (%.3f) should equal the base mean (%.3f).\n",
                     (double) apex, (double) mean );
            goto Error_cudart;
        }
        printf( "The area filter preserves the mean, as expected.\n" );
    }

    ret = 0;
Error_cudart:
    cudaFreeMipmappedArray( mipArray );
    cudaFree( devImage );
    free( host );
    return ret;
}
