/*
 *
 * corrTexConstant.cuh
 *
 * Header file for 2D implementation of normalized correlation
 * that reads the template from constant memory.
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
corrTexConstant_kernel( 
    float *pCorr, size_t CorrPitch, 
    float cPixels, float fDenomExp,
    float xUL, float yUL, int w, int h,
    int xOffset, int yOffset,
    int wTemplate, int hTemplate )
{
    size_t row = blockIdx.y*blockDim.y + threadIdx.y;
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;

    // adjust pointers to row
    pCorr = (float *) ((char *) pCorr+row*CorrPitch);

    // No __syncthreads in this kernel, so we can early-out
    // without worrying about the effects of divergence.
    if ( col >= w || row >= h )
        return;

    int SumI = 0;
    int SumISq = 0;
    int SumIT = 0;
    int inx = 0;

    for ( int j = 0; j < hTemplate; j++ ) {
        for ( int i = 0; i < wTemplate; i++ ) {
            unsigned char I = tex2D( texImage, 
                                     (float) col+xUL+xOffset+i, 
                                     (float) row+yUL+yOffset+j );
            unsigned char T = g_Tpix[inx++];
            SumI += I;
            SumISq += I*I;
            SumIT += I*T;
        }
    }
    pCorr[col] = CorrelationValue( SumI, SumISq, SumIT, g_SumT, cPixels, fDenomExp );
}

void
corrTexConstant( 
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
    corrTexConstant_kernel<<<blocks, threads>>>(
        dCorr, CorrPitch,
        cPixels, fDenomExp, 
        (float) xUL, (float) yUL, w, h,
        xOffset, yOffset, wTemplate, hTemplate );
}
