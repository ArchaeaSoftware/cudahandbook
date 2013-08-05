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
 * Requires: SM 1.1, for global atomics.
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

#include "histogramNaiveAtomic.cuh"
#include "histogramSharedAtomic.cuh"
#include "histogramSharedAtomicReduce.cuh"
#include "histogramSharedPrivatized.cuh"
#include "histogramSharedPrivatized32.cuh"

#include "histogramPrivatized8.cuh"
#include "histogramPrivatized8Pitch.cuh"
#include "histogramNPP.cuh"

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

void 
hist1DCPU( 
    unsigned int *pHist, 
    unsigned char *p, size_t N )
{
    memset( pHist, 0, 256*sizeof(int) );
    for ( size_t i = 0; i < N; i++ ) {
        pHist[ p[i] ] += 1;
    }
}

bool
TestHistogram( 
    double *pixelsPerSecond,    // passback to report performance
    const unsigned char *dptrBase, size_t dPitch,
    int w, int h,               // width and height of input
    const unsigned int *hrefHist, // host reference data
    dim3 threads, dim3 blocks,
    void (*pfnHistogram)( 
        float *ms, 
        unsigned int *pHist,
        const unsigned char *dptrBase, size_t dPitch,
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
    float ms;

    CUDART_CHECK( cudaMalloc( (void **) &dHist, 256*sizeof(int) ) );
    CUDART_CHECK( cudaMemset( dHist, 0, 256*sizeof(int) ) );

    pfnHistogram( &ms, dHist, dptrBase, dPitch, 0, 0, w, h, threads, blocks );

    CUDART_CHECK( cudaMemcpy( hHist, dHist, sizeof(hHist), cudaMemcpyDeviceToHost ) );

    if ( bCompareHistograms( hHist, hrefHist, 256 ) ) {
        printf( "Histograms miscompare\n" );
        goto Error;
    }

    for ( int i = 0; i < cIterations; i++ ) {
        pfnHistogram( &ms, dHist, dptrBase, dPitch, 0, 0, w, h, threads, blocks );
    }

    *pixelsPerSecond = (double) w*h*cIterations*1000.0 / ms;
    CUDART_CHECK( cudaMemcpy( hHist, dHist, sizeof(hHist), cudaMemcpyDeviceToHost ) );

    if ( outputFilename ) {
        FILE *f = fopen( outputFilename, "w" );
        if ( ! f )
            goto Error;
        for ( int i = 0; i < 256; i++ ) {
            fprintf( f, "%d\t", hHist[i] );
        }
        fprintf( f, "\n" );
        fclose( f );
    }

    ret = true;

Error:
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
    bool bTesla = false;

    dim3 threads;
    dim3 blocks;

    char *inputFilename = "coins.pgm";
    char *outputFilename = NULL;

    cudaArray *pArrayImage = NULL;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();

    if ( chCommandLineGetBool( "help", argc, argv ) ) {
        printf( "Usage:\n" );
        printf( "    --input <filename>: specify input filename (must be PGM)\n" );
        printf( "    --output <filename>: Write PGM of correlation values (0..255) to <filename>.\n" );
        printf( "    --padWidth <value>: pad input image width to specified value\n" );
        printf( "    --padHeight <value>: pad input image height to specified value\n" );
        printf( "    --random <numvalues>: overrides input filename and fills image with random data in the range [0..numvalues)\n" );
        printf( "    --stride <value>: specifies stride for random values (e.g., 2 means use even values only)\n" );
        printf( "    The random parameter must be in the range 1..256, and random/stride must be 256 or less.\n" );
        printf( "\nDefault values are coins.pgm and no output file or padding\n" );

        return 0;
    }

    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );
    CUDART_CHECK( cudaDeviceSetCacheConfig( cudaFuncCachePreferShared ) );

    if ( chCommandLineGet( &inputFilename, "input", argc, argv ) ) {
        printf( "Reading from image file %s\n", inputFilename );
    }
    chCommandLineGet( &outputFilename, "output", argc, argv );
    {
        int padWidth = 1024;//0;
        int padHeight = 1024;//0;
        int numvalues = 0;
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
        if ( chCommandLineGet( &numvalues, "random", argc, argv ) ) {
            int stride = 1;
            if ( chCommandLineGet( &stride, "stride", argc, argv ) ) {
                if ( numvalues*stride > 256 ) {
                    printf( "stride*random must be <= 256\n" );
                    goto Error;
                }
            }
            if ( 0==padWidth || 0==padHeight ) {
                printf( "--random requires --padWidth and padHeight (to specify input size)\n" );
                goto Error;
            }
            printf( "%d pixels, random, %d values with stride %d\n",
                padWidth*padHeight, numvalues, stride );
            w = padWidth;
            h = padWidth;
            hidata = (unsigned char *) malloc( w*h );
            if ( ! hidata )
                goto Error;

            size_t dPitch;
            CUDART_CHECK( cudaMallocPitch( &didata, &dPitch, padWidth, padHeight ) );
            DevicePitch = dPitch;

            srand(42);
            for ( int row = 0; row < h; row++ ) {
                unsigned char *p = hidata+row*w;
                for ( int col = 0; col < w; col++ ) {
                    int val = rand() % numvalues;
                    val *= stride;
                    p[col] = (unsigned char) val;
                }
            }
            CUDART_CHECK( cudaMemcpy2D( didata, DevicePitch, hidata, padWidth, padWidth, padHeight, cudaMemcpyHostToDevice ) );
        }
        else {
            if ( pgmLoad( inputFilename, &hidata, &HostPitch, &didata, &DevicePitch, &w, &h, padWidth, padHeight) )
                goto Error;
             printf( "%d pixels, sourced from image file %s\n", w*h, inputFilename );
        }
    }


    CUDART_CHECK( cudaMallocArray( &pArrayImage, &desc, w, h ) );
    CUDART_CHECK( cudaMemcpyToArray( pArrayImage, 0, 0, hidata, w*h, cudaMemcpyHostToDevice ) );
        
    CUDART_CHECK( cudaBindTextureToArray( texImage, pArrayImage ) );

    {
        cudaDeviceProp prop;
        CUDART_CHECK( cudaGetDeviceProperties( &prop, 0 ) );
        if ( prop.major < 2 ) {
            bTesla = true;
        }
    }

    histCPU( cpuHist, w, h, hidata, w );
    {
        unsigned int cpuHist2[256];
        hist1DCPU( cpuHist2, hidata, w*h );
        if ( bCompareHistograms( cpuHist, cpuHist2, 256 ) ) {
            printf( "Linear and 2D histograms do not agree\n" );
            exit(1);
        }
    }
    

#define TEST_VECTOR( baseName, bPrintNeighborhood, cIterations, outfile ) \
    { \
        double pixelsPerSecond; \
        if ( ! TestHistogram( &pixelsPerSecond, \
            didata, DevicePitch, \
            w, h,  \
            cpuHist, \
            threads, blocks,  \
            baseName, \
            cIterations, outfile ) ) { \
            printf( "Error\n" ); \
            ret = 1; \
            goto Error; \
        } \
        printf( "%s: %.2f Mpix/s\n", \
            #baseName, pixelsPerSecond/1e6 ); \
    }

    if ( w != DevicePitch ) {
        printf( "1D versions only work if width and pitch are the same\n" );
    }

    threads = dim3( 32, 8, 1 );
    blocks = dim3( 40, 40, 1 );

    TEST_VECTOR( GPUhistogramNaiveAtomic, false, 1, NULL );
    threads = dim3( 16, 4, 1 );
    TEST_VECTOR( GPUhistogramSharedAtomic, false, 1, NULL );
    TEST_VECTOR( GPUhistogramSharedAtomicReduce, false, 1, NULL );
    TEST_VECTOR( GPUhistogramSharedAtomic4x, false, 1, NULL );
    threads = dim3( 16, 4, 1 );
    if ( ! bTesla ) {
        TEST_VECTOR( GPUhistogramSharedPrivatized, false, 1, NULL );
        TEST_VECTOR( GPUhistogramSharedPrivatized32, false, 1, NULL );

        TEST_VECTOR( GPUhistogramPrivatized8, false, 1, NULL );
        TEST_VECTOR( GPUhistogramPrivatized8Pitch, false, 1, NULL );
    }

    TEST_VECTOR( GPUhistogramNPP, false, 1, NULL );

    ret = 0;
Error:
    free( hidata );
    cudaFree(didata); 

    cudaFreeArray(pArrayImage);
   
    return ret;
}
