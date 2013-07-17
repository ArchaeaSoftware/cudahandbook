/*
 *
 * pgm.cu
 *
 * Functions to load and store PGM (portable gray map) files.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

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
#include <string.h>
#include <assert.h>
#include "pgm.h"

int
pgmLoad(
    const char *filename, 
    unsigned char **pHostData, unsigned int *pHostPitch, 
    unsigned char **pDeviceData, unsigned int *pDevicePitch,
    int *pWidth, int *pHeight, int padWidth, int padHeight)
{
    int ret = 1;
    const int hsize = 0x40;
    int w, h;
    FILE *fp = NULL;
    int maxval;
    char header[hsize];
    unsigned char *idata = NULL;
    unsigned char *ddata = NULL;
    size_t dPitch;

    fp = fopen( filename, "rb" );
    if ( fp == NULL) {
        fprintf( stderr, "Failed to open %s.\n", filename );
        goto Error;
    }

    if (NULL == fgets(header, hsize, fp)) {
        fprintf(stderr, "Invalid PGM file.\n");
        goto Error;
    }

    if ( strncmp(header, "P5", 2) ) {
        fprintf(stderr, "File is not a PGM image.\n");
        goto Error;
    }
    if ( 1 != fscanf( fp, "%d", &w ) )
        goto Error;
    if ( 1 != fscanf( fp, "%d", &h ) )
        goto Error;
    if ( 1 != fscanf( fp, "%d", &maxval ) )
        goto Error;
    if ( padWidth == 0 && padHeight == 0 ) {
        padWidth = w;
        padHeight = h;
    }
    idata = (unsigned char *) malloc( padWidth*padHeight );
    if ( ! idata )
        goto Error;
    for ( int row = 0; row < h; row++ ) {
        if ( (size_t) w != fread( idata+row*padWidth, 1, w, fp ) )
            goto Error;
    }
    if ( cudaSuccess != cudaMallocPitch( (void **) &ddata, &dPitch, padWidth, padHeight ) )
        goto Error;
    *pWidth = padWidth;
    *pHeight = padHeight;
    *pHostPitch = padWidth;
    *pHostData = idata;
    *pDeviceData = ddata;
    *pDevicePitch = (unsigned int) dPitch;
    cudaMemcpy2D( ddata, dPitch, idata, padWidth, padWidth, padHeight, cudaMemcpyHostToDevice );
    fclose(fp);
    return 0;
Error:
    free( idata );
    cudaFree( ddata );
    if ( fp ) {
        fclose( fp );
    }
    return ret;
}

int
pgmSave(const char* filename, unsigned char *data, int w, int h)
{
    int ret = 1;
    FILE *fp = fopen( filename, "wb" );
    if ( NULL == fp ) {
        fprintf( stderr, "Failed to open %s\n", filename );
        goto Error;
    }

    fprintf( fp, "P5\n%d\n%d\n%d\n", w, h, 0xff );
    if ( w*h != fwrite(data, sizeof(unsigned char), w*h, fp) ) {
        fprintf( stderr, "Write failed\n" );
        goto Error;
    }

    fclose(fp);
    ret = 0;
Error:
    return ret;
}
