/*
 *
 * normalizedCrossCorrelation.cu
 *
 * Microbenchmark for normalized cross correlation, a template-
 * matching algorithm for computer vision.
 *
 * Build with: nvcc -I ../chLib <options> normalizedCrossCorrelation.cu ..\chLib\pgm.cu
 *
 * Make sure to include pgm.cu for the image file I/O support.
 *
 * To avoid warnings about double precision support, specify the
 * target gpu-architecture, e.g.:
 * nvcc --gpu-architecture sm_13 -I ../chLib <options> normalizedCrossCorrelation.cu pgm.cu
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
texture<unsigned char, 2> texTemplate;

const int maxTemplatePixels = 3072;
__constant__ int g_xOffset[maxTemplatePixels];
__constant__ int g_yOffset[maxTemplatePixels];
__constant__ unsigned char g_Tpix[maxTemplatePixels];
__constant__ float g_cPixels, g_SumT, g_fDenomExp;
unsigned int g_cpuSumT, g_cpuSumTSq;

const float fThreshold = 1e-3f;

#define INTCEIL(a,b) ( ((a)+(b)-1) / (b) )

__device__ __host__ inline float
CorrelationValue( float SumI, float SumISq, float SumIT, float SumT, float cPixels, float fDenomExp )
{
    float Numerator = cPixels*SumIT - SumI*SumT;
    float Denominator = rsqrtf( (cPixels*SumISq - SumI*SumI)*fDenomExp );
    return Numerator * Denominator;
}

#include "corrTexTexSums.cuh"
#include "corrTexTex.cuh"

#include "corrTexConstantSums.cuh"
#include "corrTexConstant.cuh"

extern __shared__ unsigned char LocalBlock[];

#include "corrSharedSMSums.cuh"
#include "corrSharedSM.cuh"

#include "corrSharedSums.cuh"
#include "corrShared.cuh"

#include "corrShared4Sums.cuh"
#include "corrShared4.cuh"

int poffsetx[maxTemplatePixels];
int poffsety[maxTemplatePixels];

cudaError_t
CopyToTemplate( 
      unsigned char *img, size_t imgPitch, 
      int xTemplate, int yTemplate,
      int wTemplate, int hTemplate,
      int OffsetX, int OffsetY
)
{
    cudaError_t status;
    unsigned char pixels[maxTemplatePixels];

    int inx = 0;
    int SumT = 0;
    int SumTSq = 0;
    int cPixels = wTemplate*hTemplate;
    size_t sizeOffsets = cPixels*sizeof(int);
    float fSumT, fDenomExp, fcPixels;

    cuda(Memcpy2D( 
        pixels, wTemplate,
        img+yTemplate*imgPitch+xTemplate, imgPitch,
        wTemplate, hTemplate,
        cudaMemcpyDeviceToHost ) );

    cuda(MemcpyToSymbol( g_Tpix, pixels, cPixels ) );

    for ( int i = OffsetY; i < OffsetY+hTemplate; i++ ) {
        for ( int j = OffsetX; j < OffsetX+wTemplate; j++) {
            SumT += pixels[inx];
            SumTSq += pixels[inx]*pixels[inx];
            poffsetx[inx] = j;
            poffsety[inx] = i;
            inx += 1;
        }
    }
    g_cpuSumT = SumT;
    g_cpuSumTSq = SumTSq;

    cuda(MemcpyToSymbol(g_xOffset, poffsetx, sizeOffsets) );
    cuda(MemcpyToSymbol(g_yOffset, poffsety, sizeOffsets) );

    fSumT = (float) SumT;
    cuda(MemcpyToSymbol(g_SumT, &fSumT, sizeof(float)) );

    fDenomExp = float( (double)cPixels*SumTSq - (double) SumT*SumT);
    cuda(MemcpyToSymbol(g_fDenomExp, &fDenomExp, sizeof(float)) );

    fcPixels = (float) cPixels;
    cuda(MemcpyToSymbol(g_cPixels, &fcPixels, sizeof(float)) );
Error:
    return status;
}

int
bCompareCorrValues( const float *pBase0, 
                    const float *pBase1, 
                    int w, int h )
{
    for ( int j = 0; j < h; j++ ) {

        float *pf0 = (float *) ((char *) pBase0+j*w*sizeof(float));
        float *pf1 = (float *) ((char *) pBase1+j*w*sizeof(float));

        for ( int i = 0; i < w; i++ ) {
            if ( fabsf(pf0[i]-pf1[i]) > fThreshold ) { 
                printf( "Mismatch pf0[%d] = %.5f, pf1[%d] = %.5f\n", i, pf0[i], i, pf1[i] ); 
                fflush( stdout );
                //CH_ASSERT(0);
                return 1;
            }
        }
    }
    return 0;
}

int
bCompareSums( const int *pBaseI0, const int *pBaseISq0, const int *pBaseIT0,
              const int *pBaseI1, const int *pBaseISq1, const int *pBaseIT1,
              int w, int h )
{
    for ( int j = 0; j < h; j++ ) {

        const int *pi0 = (const int *) ((char *) pBaseI0+j*w*sizeof(int));
        const int *pi1 = (const int *) ((char *) pBaseI1+j*w*sizeof(int));

        const int *pisq0 = (const int *) ((char *) pBaseISq0+j*w*sizeof(int));
        const int *pisq1 = (const int *) ((char *) pBaseISq1+j*w*sizeof(int));

        const int *pit0 = (const int *) ((char *) pBaseIT0+j*w*sizeof(int));
        const int *pit1 = (const int *) ((char *) pBaseIT1+j*w*sizeof(int));
        for ( int i = 0; i < w; i++ ) {
            if ( pi0[i] != pi1[i] ||
                 pisq0[i] != pisq1[i] ||
                 pit0[i] != pit1[i] ) { 
                printf( "Mismatch pi[%d] = %d, reference = %d\n", i, pi0[i], pi1[i] ); 
                printf( "Mismatch pisq[%d] = %d, reference = %d\n", i, pisq0[i], pisq1[i] );
                printf( "Mismatch pit[%d] = %d, reference = %d\n", i, pit0[i], pit1[i] );
                fflush( stdout );
                //CH_ASSERT(0);
                return 1;
            }
        }
    }
    return 0;
}

unsigned char
ReadPixel( unsigned char *base, int pitch, int w, int h, int x, int y )
{
    if ( x < 0 ) x = 0;
    if ( x >= w ) x = w-1;
    if ( y < 0 ) y = 0;
    if ( y >= h ) y = h-1;
    return base[y*pitch+x];
}

void 
corrCPU( float *pCorr, 
         int *_pI, int *_pISq, int *_pIT,
         size_t CorrPitch, 
         int cPixels,
         int xTemplate, int yTemplate,
         int w, int h,
         unsigned char *img, int imgPitch,
         unsigned char *tmp, int tmpPitch )
{
    for ( int row = 0; row < h; row += 1 ) {
        float *pOut = (float *) (((char *) pCorr)+row*CorrPitch);
        int *pI = (int *) (((char *) _pI)+row*CorrPitch);
        int *pISq = (int *) (((char *) _pISq)+row*CorrPitch);
        int *pIT = (int *) (((char *) _pIT)+row*CorrPitch);
        for ( int col = 0; col < w; col += 1 ) {
            int SumI = 0;
            int SumT = 0;
            int SumISq = 0;
            int SumTSq = 0;
            int SumIT = 0;
            for ( int j = 0; j < cPixels; j++ ) {
                unsigned char I = ReadPixel( img, imgPitch, w, h, col+poffsetx[j], row+poffsety[j] );
                unsigned char T = ReadPixel( tmp, tmpPitch, w, h, xTemplate+poffsetx[j], yTemplate+poffsety[j] );
                SumI += I;
                SumT += T;
                SumISq += I*I;
                SumTSq += T*T;
                SumIT += I*T;
            }
            float fDenomExp = float((double) cPixels*SumTSq - (double) SumT*SumT);
            pI[col] = SumI;
            pISq[col] = SumISq;
            pIT[col] = SumIT;
            pOut[col] = CorrelationValue( (float) SumI, (float) SumISq, (float) SumIT, (float) SumT, (float) cPixels, fDenomExp );
        }
    }
}

bool
TestCorrelation( 
    double *pixelsPerSecond,         // passbacks to report performance
    double *templatePixelsPerSecond, // 
    int xOffset, int yOffset,  // offset into image
    int w, int h,              // width and height of output
    const float *hrefCorr,     // host reference data
    const int *hrefSumI,
    const int *hrefSumISq, 
    const int *hrefSumIT,
    int xTemplate, int yTemplate, // reference point in template image
    int wTemplate, int hTemplate,
    int wTile,                 // width of image tile
    int sharedPitch, int sharedMem,
    dim3 threads, dim3 blocks,
    void (*pfnCorrelationSums)( 
        float *dCorr, int CorrPitch,
        int *dSumI, int *dSumISq, int *dSumIT,
        int wTile,
        int wTemplate, int hTemplate,
        float cPixels,
        float fDenomExp,
        int sharedPitch,
        int xOffset, int yOffset,
        int xTemplate, int yTemplate,
        int xUL, int yUL, int w, int h,
        dim3 threads, dim3 blocks,
        int sharedMem ),
    void (*pfnCorrelation)( 
        float *dCorr, int CorrPitch,
        int wTile,
        int wTemplate, int hTemplate,
        float cPixels,
        float fDenomExp,
        int sharedPitch,
        int xOffset, int yOffset,
        int xTemplate, int yTemplate,
        int xUL, int yUL, int w, int h,
        dim3 threads, dim3 blocks,
        int sharedMem ),
    bool bPrintNeighborhood = false,
    int cIterations = 1,
    const char *outputFilename = NULL
)
{
    cudaError_t status;
    bool ret = false;
    size_t CorrPitch;

    float cPixels = (float) wTemplate*hTemplate;
    float fDenomExp = float((double) cPixels*g_cpuSumTSq - (double) g_cpuSumT*g_cpuSumT);

    float *hCorr = NULL, *dCorr = NULL;
    int *hSumI = NULL, *dSumI = NULL;
    int *hSumISq = NULL, *dSumISq = NULL;
    int *hSumIT = NULL, *dSumIT = NULL;

    cudaEvent_t start = 0, stop = 0;

    hCorr = (float *) malloc( w*sizeof(float)*h );
    hSumI = (int *) malloc( w*sizeof(int)*h );
    hSumISq = (int *) malloc( w*sizeof(int)*h );
    hSumIT = (int *) malloc( w*sizeof(int)*h );
    if ( NULL == hCorr || NULL == hSumI || NULL == hSumISq || NULL == hSumIT )
        goto Error;

    cuda(MallocPitch( (void **) &dCorr, &CorrPitch, w*sizeof(float), h ) );
    cuda(MallocPitch( (void **) &dSumI, &CorrPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumISq, &CorrPitch, w*sizeof(int), h ) );
    cuda(MallocPitch( (void **) &dSumIT, &CorrPitch, w*sizeof(int), h ) );

    cuda(Memset( dCorr, 0, CorrPitch*h ) );
    cuda(Memset( dSumI, 0, CorrPitch*h ) );
    cuda(Memset( dSumISq, 0, CorrPitch*h ) );
    cuda(Memset( dSumIT, 0, CorrPitch*h ) );

    cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );

    pfnCorrelationSums(
        dCorr, CorrPitch,
        dSumI, dSumISq, dSumIT,
        wTile,
        wTemplate, hTemplate, 
        cPixels, fDenomExp, 
        sharedPitch, 
        xOffset, yOffset, 
        xTemplate, yTemplate, 
        0, 0, w, h,
        threads, blocks, sharedMem );

    cuda(Memcpy2D( hSumI, w*sizeof(int), dSumI, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumISq, w*sizeof(int), dSumISq, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );
    cuda(Memcpy2D( hSumIT, w*sizeof(int), dSumIT, CorrPitch, w*sizeof(int), h, cudaMemcpyDeviceToHost ) );

    if ( bCompareSums( hSumI, hSumISq, hSumIT,
                       hrefSumI, hrefSumISq, hrefSumIT,
                       w, h ) ) {
        //CH_ASSERT(0);
        printf( "Sums miscompare\n" );
        goto Error;
    }

    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );

    if ( bCompareCorrValues( hrefCorr, hCorr, w, h ) ) {
        //CH_ASSERT(0);
        printf( "Correlation coefficients generated by sums kernel mismatch\n" );
        return 1;
    }

    cuda(Memset2D( dCorr, CorrPitch, 0, w*sizeof(float), h ) );
    cuda(DeviceSynchronize() );
    cuda(EventRecord( start, 0 ) );

    for ( int i = 0; i < cIterations; i++ ) {
        pfnCorrelation( 
            dCorr, CorrPitch, 
            wTile, 
            wTemplate, hTemplate, 
            cPixels, fDenomExp, 
            sharedPitch, 
            xOffset, yOffset,
            xTemplate, yTemplate, 
            0, 0, w, h, 
            threads, blocks, sharedMem );
    }

    cuda(EventRecord( stop, 0 ) );
    cuda(Memcpy2D( hCorr, w*sizeof(float), dCorr, CorrPitch, w*sizeof(float), h, cudaMemcpyDeviceToHost ) );

    if ( bCompareCorrValues( hrefCorr, hCorr, w, h ) ) {
        CH_ASSERT(0);
        printf( "Correlation coefficients generated by coefficient-only kernel mismatch\n" );
        return 1;
    }

    {
        float ms;
        cuda(EventElapsedTime( &ms, start, stop ) );
        *pixelsPerSecond = (double) w*h*cIterations*1000.0 / ms;
        *templatePixelsPerSecond = *pixelsPerSecond*wTemplate*hTemplate;
    }

    if ( bPrintNeighborhood ) {
        printf( "\nNeighborhood around template:\n" );
        for ( int VertOffset = -4; VertOffset <= 4; VertOffset++ ) {
            const float *py = hrefCorr+w*(VertOffset+yTemplate);
            for ( int HorzOffset = -4; HorzOffset <= 4; HorzOffset++ ) {
                printf( "%6.2f", py[xTemplate+HorzOffset] );
            }
            printf("\n");
        }
    }

    if ( outputFilename ) {
        unsigned char *correlationValues = (unsigned char *) malloc( w*h );
        if ( ! correlationValues ) {
            status = cudaErrorMemoryAllocation;
            goto Error;
        }
        for ( int row = 0; row < h; row++ ) {
            for ( int col = 0; col < w; col++ ) {
                int index = row*w+col;
                float value = hCorr[index] < 0.0f ? 0.0f : logf( 1.0f+hCorr[index] )/logf(2.0f);
                if ( value < 0.5f ) value = 0.0f;
                value = 2.0f * (value - 0.5f);
                correlationValues[index] = (unsigned char) (255.0f*value+0.5f);
            }
        }
        if ( 0 != pgmSave( outputFilename, correlationValues, w, h ) ) {
            status = cudaErrorUnknown;
            goto Error;
        }
        free( correlationValues );
    }

    ret = true;

Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    free( hCorr );
    free( hSumI );
    free( hSumISq );
    free( hSumIT );
    if ( dCorr ) cudaFree( dCorr );
    if ( dSumI ) cudaFree( dSumI );
    if ( dSumI ) cudaFree( dSumISq );
    if ( dSumI ) cudaFree( dSumIT );
    return ret;
}

int
main(int argc, char *argv[])
{
    int ret = 1;
    cudaError_t status;

    unsigned char *hidata = NULL;
    unsigned char *didata = NULL;
    float *hoCorrCPU = NULL;
    
    int *hoCorrCPUI = NULL;
    int *hoCorrCPUISq = NULL;
    int *hoCorrCPUIT = NULL;
    unsigned int HostPitch, DevicePitch;
    int w, h;

    int wTemplate = 52;
    int hTemplate = 52;
    int xOffset, yOffset;

    int xTemplate = 210;
    int yTemplate = 148;

    int wTile;
    dim3 threads;
    dim3 blocks;

    int sharedPitch;
    int sharedMem;
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

    cuda(SetDeviceFlags( cudaDeviceMapHost ) );
    cuda(DeviceSetCacheConfig( cudaFuncCachePreferShared ) );

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
    chCommandLineGet( &xTemplate, "xTemplate", argc, argv );
    chCommandLineGet( &yTemplate, "yTemplate", argc, argv );
    chCommandLineGet( &wTemplate, "wTemplate", argc, argv );
    chCommandLineGet( &hTemplate, "hTemplate", argc, argv );

    xOffset = -wTemplate/2;
    yOffset = -wTemplate/2;

    hoCorrCPU = (float *) malloc(w*h*sizeof(float)); if ( ! hoCorrCPU ) return 1;
    hoCorrCPUI = (int *) malloc(w*h*sizeof(int)); if ( ! hoCorrCPUI ) return 1;
    hoCorrCPUISq = (int *) malloc(w*h*sizeof(int)); if ( ! hoCorrCPUISq ) return 1;
    hoCorrCPUIT = (int *) malloc(w*h*sizeof(int)); if ( ! hoCorrCPUIT ) return 1;
    if ( NULL == hoCorrCPU ||
         NULL == hoCorrCPUI ||
         NULL == hoCorrCPUISq ||
         NULL == hoCorrCPUIT )
        goto Error;

    cuda(MallocArray( &pArrayImage, &desc, w, h ) );
    cuda(MallocArray( &pArrayTemplate, &desc, w, h ) );
    cuda(MemcpyToArray( pArrayImage, 0, 0, hidata, w*h, cudaMemcpyHostToDevice ) );
        
    cuda(Memcpy2DArrayToArray( pArrayTemplate, 0, 0, pArrayImage, 0, 0, w, h, cudaMemcpyDeviceToDevice ) );
    
    cuda(BindTextureToArray( texImage, pArrayImage ) );
    cuda(BindTextureToArray( texTemplate, pArrayTemplate ) );

    CopyToTemplate( didata, DevicePitch, 
                    xTemplate, yTemplate, 
                    wTemplate, hTemplate,
                    xOffset, yOffset );

    corrCPU( hoCorrCPU, hoCorrCPUI, hoCorrCPUISq, hoCorrCPUIT, 
        w*sizeof(float), wTemplate*hTemplate, xTemplate-xOffset, yTemplate-yOffset, w, h, 
        hidata, HostPitch, hidata, HostPitch );

    // height of thread block must be >= hTemplate
    wTile = 32;
    threads = dim3(32,8);
    blocks = dim3(w/wTile+(0!=w%wTile),h/threads.y+(0!=h%threads.y));

    sharedPitch = ~63&(wTile+wTemplate+63);
    sharedMem = sharedPitch*(threads.y+hTemplate);

#define TEST_VECTOR( baseName, bPrintNeighborhood, cIterations, outfile ) \
    { \
        double pixelsPerSecond; \
        double templatePixelsPerSecond; \
        if ( ! TestCorrelation( &pixelsPerSecond, \
            &templatePixelsPerSecond, \
            xOffset, yOffset, \
            w, h,  \
            hoCorrCPU, \
            hoCorrCPUI, \
            hoCorrCPUISq, \
            hoCorrCPUIT, \
            xTemplate-xOffset, yTemplate-yOffset, \
            wTemplate, hTemplate, \
            wTile, sharedPitch, sharedMem, \
            threads, blocks,  \
            baseName##Sums,  \
            baseName, \
            bPrintNeighborhood, cIterations, outfile ) ) { \
            printf( "Error\n" ); \
        } \
        printf( "%s: %.2f Mpix/s\t%.2fGtpix/s\n", \
            #baseName, pixelsPerSecond/1e6, templatePixelsPerSecond/1e9 ); \
    }

    TEST_VECTOR( corrShared, false, 100, NULL );

    // height of thread block must be >= hTemplate
    wTile = 32;
    threads = dim3(32,8);
    blocks = dim3(w/wTile+(0!=w%wTile),h/threads.y+(0!=h%threads.y));

    sharedPitch = ~63&(((wTile+wTemplate)+63));
    sharedMem = sharedPitch*(threads.y+hTemplate);

    TEST_VECTOR( corrSharedSM, false, 100, NULL );

    TEST_VECTOR( corrShared4, false, 100, NULL );

    // set up blocking parameters for 2D tex-constant formulation
    threads.x = 32; threads.y = 16; threads.z = 1;
    blocks.x = INTCEIL(w,threads.x); blocks.y = INTCEIL(h,threads.y); blocks.z = 1;
    TEST_VECTOR( corrTexConstant, false, 100, NULL );

    if ( outputFilename ) {
        printf( "Writing graymap of correlation values to %s\n", outputFilename );
    }

    // set up blocking parameters for 2D tex-tex formulation
    threads.x = 16; threads.y = 8; threads.z = 1;
    blocks.x = INTCEIL(w,threads.x); blocks.y = INTCEIL(h,threads.y); blocks.z = 1;
    TEST_VECTOR( corrTexTex, false, 100, outputFilename );

    ret = 0;
Error:
    free( hoCorrCPU );
    free( hoCorrCPUI );
    free( hoCorrCPUISq );
    free( hoCorrCPUIT );

    free( hidata );

    cudaFree(didata); 

    cudaFreeArray(pArrayImage);
    cudaFreeArray(pArrayTemplate);
   
    return ret;

}
