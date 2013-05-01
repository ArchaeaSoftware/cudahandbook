/*
 *
 * corrTexTex.cuh
 *
 * Header file for implementation of normalized correlation
 * that reads both image and template from texture.
 *
 * Copyright (c) 2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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

__global__ void 
corrTexTex_kernel( 
    float *pCorr, size_t CorrPitch, 
    float cPixels,
    int xOffset, int yOffset,
    int xTemplate, int yTemplate,
    int wTemplate, int hTemplate,
    float xUL, float yUL, int w, int h )
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

    // adjust pCorr to point to row
    pCorr = (float *) ((char *) pCorr+row*CorrPitch);

    // No __syncthreads in this kernel, so we can early-out
    // without worrying about the effects of divergence.
    if ( col >= w || row >= h )
        return;

    int SumI = 0;
    int SumT = 0;
    int SumISq = 0;
    int SumTSq = 0;
    int SumIT = 0;
    for ( int y = 0; y < hTemplate; y++ ) {
        for ( int x = 0; x < wTemplate; x++ ) {
            unsigned char I = tex2D( texImage, 
                (float) col+xUL+xOffset+x, (float) row+yUL+yOffset+y );
            unsigned char T = tex2D( texTemplate, 
                (float) xTemplate+x, (float) yTemplate+y);
            SumI += I;
            SumT += T;
            SumISq += I*I;
            SumTSq += T*T;
            SumIT += I*T;
        }
        float fDenomExp = float( (double) cPixels*SumTSq - (double) SumT*SumT);
        pCorr[col] = CorrelationValue( SumI, SumISq, SumIT, SumT, cPixels, fDenomExp );
    }
}

void
corrTexTex( 
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
    int sharedMem )
{
    corrTexTex_kernel<<<blocks, threads>>>( 
        dCorr, CorrPitch,
        cPixels,
        xOffset, yOffset,
        xTemplate+xOffset, yTemplate+yOffset,
        wTemplate, hTemplate,
        (float) xUL, (float) yUL, w, h );
}
