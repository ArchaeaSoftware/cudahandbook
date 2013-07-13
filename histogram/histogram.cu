/*
 *
 * histogram.cu
 *
 * Microbenchmark for histogram, a statistical computation
 * for image processing.
 *
 * Build with: nvcc -I ../chLib <options> histogram.cu ..\chLib\pgm.cu
 *
 * Make sure to include pgm.cu for the image file I/O support.
 *
 * To avoid warnings about double precision support, specify the
 * target gpu-architecture, e.g.:
 * nvcc --gpu-architecture sm_13 -I ../chLib <options> histogram.cu ..\chLib\pgm.cu
 *
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

#include <chError.h>
#include <chCommandLine.h>
#include <chAssert.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "pgm.h"

texture<unsigned char, 2> texImage;

__global__ void
histogramNaiveAtomic( 
    unsigned int *pHist, 
    int x, int y, 
    int w, int h )
{
    for ( int row = blockIdx.y*blockDim.y+threadIdx.y; 
              row < h;
              row += blockDim.y*gridDim.y ) {
        for ( int col = blockIdx.x*blockDim.x+threadIdx.x;
                  col < w;
                  col += blockDim.x*gridDim.x ) {
            unsigned char pixval = tex2D( texImage, (float) col, (float) row );
            atomicAdd( &pHist[pixval], 1 );
        }
    }
}

void
GPUhistogramNaiveAtomic(
    unsigned int *pHist,
    int x, int y,
    int w, int h, 
    dim3 threads, dim3 blocks )
{
    histogramNaiveAtomic<<<blocks,threads>>>( pHist, x, y, w, h );
}
    

int
bCompareHistograms( const unsigned int *p, const unsigned int *q, int N )
{
    for ( int i = 0; i < N; i++ ) {
        if ( p[i] != q[i] ) {
            printf( "Histogram mismatch at %d: p[%d] == %d, q[%d] == %d\n", i, i, p[i], i, q[i] );
            return 1;
        }
    }
    return 0;
}

void 
histCPU( 
    unsigned int *pHist, 
    int w, int h,
    unsigned char *img, int imgPitch )
{
    memset( pHist, 0, 256*sizeof(int) );
    for ( int row = 0; row < h; row += 1 ) {
        unsigned char *pi = img+row*imgPitch;
        for ( int col = 0; col < w; col += 1 ) {
            pHist[pi[col]] += 1;
        }
    }
}

bool
TestHistogram( 
    double *pixelsPerSecond,    // passback to report performance
    int w, int h,               // width and height of input
    const unsigned int *hrefHist, // host reference data
    dim3 threads, dim3 blocks,
    void (*pfnHistogram)( 
        unsigned int *pHist,
        int xUL, int yUL, int w, int h,
        dim3 threads, dim3 blocks ),
    int cIterations = 1,
    const char *outputFilename = NULL
)
{
    cudaError_t status;
    bool ret = false;

    // Histogram for 8-bit grayscale image (2^8=256)
    unsigned int hHist[256];
    
    unsigned int *dHist = NULL;

    cudaEvent_t start = 0, stop = 0;

    CUDART_CHECK( cudaMalloc( (void **) &dHist, 256*sizeof(int) ) );
    CUDART_CHECK( cudaMemset( dHist, 0, 256*sizeof(int) ) );

    CUDART_CHECK( cudaEventCreate( &start, 0 ) );
    CUDART_CHECK( cudaEventCreate( &stop, 0 ) );

    pfnHistogram( dHist, 0, 0, w, h, threads, blocks );

    CUDART_CHECK( cudaMemcpy( hHist, dHist, sizeof(hHist), cudaMemcpyDeviceToHost ) );

    if ( bCompareHistograms( hHist, hrefHist, 256 ) ) {
        //CH_ASSERT(0);
        printf( "Sums miscompare\n" );
        goto Error;
    }

    CUDART_CHECK( cudaEventRecord( start, 0 ) );

    for ( int i = 0; i < cIterations; i++ ) {
        pfnHistogram( dHist, 0, 0, w, h, threads, blocks );
    }

    CUDART_CHECK( cudaEventRecord( stop, 0 ) );

    CUDART_CHECK( cudaMemcpy( hHist, dHist, sizeof(hHist), cudaMemcpyDeviceToHost ) );

    if ( bCompareHistograms( hHist, hrefHist, 256 ) ) {
        //CH_ASSERT(0);
        printf( "Sums miscompare\n" );
        goto Error;
    }

    {
        float ms;
        CUDART_CHECK( cudaEventElapsedTime( &ms, start, stop ) );
        *pixelsPerSecond = (double) w*h*cIterations*1000.0 / ms;
    }

    ret = true;

Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaFree( dHist );
    return ret;
}

int
main(int argc, char *argv[])
{
    int ret = 1;
    cudaError_t status;

    unsigned char *hidata = NULL;
    unsigned char *didata = NULL;
    
    unsigned int cpuHist[256];
    unsigned int HostPitch, DevicePitch;
    int w, h;

    dim3 threads;
    dim3 blocks;

    char *inputFilename = "coins.pgm";
    char *outputFilename = NULL;

    cudaArray *pArrayImage = NULL;
    cudaArray *pArrayTemplate = NULL;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();

    if ( chCommandLineGetBool( "help", argc, argv ) ) {
        printf( "Usage:\n" );
        printf( "    --input <filename>: specify input filename (must be PGM)\n" );
        printf( "    --output <filename>: Write PGM of correlation values (0..255) to <filename>.\n" );
        printf( "    --padWidth <value>: pad input image width to specified value\n" );
        printf( "    --padHeight <value>: pad input image height to specified value\n" );
        printf( "    --xTemplate <value>: X coordinate of upper left corner of template\n" );
        printf( "    --yTemplate <value>: Y coordinate of upper left corner of template\n" );
        printf( "    --wTemplate <value>: Width of template\n" );
        printf( "    --hTemplate <value>: Height of template\n" );
        printf( "\nDefault values are coins.pgm, no output file or padding, and template of the dime in the\n" );
        printf("lower right corner of coins.pgm: xTemplate=210, yTemplate=148, wTemplate=hTemplate=52\n" );

        return 0;
    }

    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    CUDART_CHECK( cudaDeviceSetCacheConfig( cudaFuncCachePreferShared ) );

    if ( chCommandLineGet( &inputFilename, "input", argc, argv ) ) {
        printf( "Reading from image file %s\n", inputFilename );
    }
    chCommandLineGet( &outputFilename, "output", argc, argv );
    {
        int padWidth = 0;
        int padHeight = 0;
        if ( chCommandLineGet( &padWidth, "padWidth", argc, argv ) ) {
            if ( ! chCommandLineGet( &padHeight, "padHeight", argc, argv ) ) {
                printf( "Must specify both --padWidth and --padHeight\n" );
                goto Error;
            }
        }
        else {
            if ( chCommandLineGet( &padHeight, "padHeight", argc, argv ) ) {
                printf( "Must specify both --padWidth and --padHeight\n" );
                goto Error;
            }
        }
        if ( pgmLoad(inputFilename, &hidata, &HostPitch, &didata, &DevicePitch, &w, &h, padWidth, padHeight) )
            goto Error;
    }

    CUDART_CHECK( cudaMallocArray( &pArrayImage, &desc, w, h ) );
    CUDART_CHECK( cudaMallocArray( &pArrayTemplate, &desc, w, h ) );
    CUDART_CHECK( cudaMemcpyToArray( pArrayImage, 0, 0, hidata, w*h, cudaMemcpyHostToDevice ) );
        
    CUDART_CHECK( cudaMemcpy2DArrayToArray( pArrayTemplate, 0, 0, pArrayImage, 0, 0, w, h, cudaMemcpyDeviceToDevice ) );
    
    CUDART_CHECK( cudaBindTextureToArray( texImage, pArrayImage ) );

    histCPU( cpuHist, w, h, hidata, w );

#define TEST_VECTOR( baseName, bPrintNeighborhood, cIterations, outfile ) \
    { \
        double pixelsPerSecond; \
        if ( ! TestHistogram( &pixelsPerSecond, \
            w, h,  \
            cpuHist, \
            threads, blocks,  \
            baseName, \
            cIterations, outfile ) ) { \
            printf( "Error\n" ); \
        } \
        printf( "%s: %.2f Mpix/s\n", \
            #baseName, pixelsPerSecond/1e6 ); \
    }

    threads = dim3( 16, 4, 1 );
    blocks = dim3( 40, 40, 1 );

    TEST_VECTOR( GPUhistogramNaiveAtomic, false, 100, NULL );

    ret = 0;
Error:
    free( hidata );
    cudaFree(didata); 

    cudaFreeArray(pArrayImage);
    cudaFreeArray(pArrayTemplate);
   
    return ret;

}
