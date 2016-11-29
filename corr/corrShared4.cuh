/*
 *
 * corrShared.cuh
 *
 * Header file for implementation of normalized correlation
 * that reads the image from shared memory and the template 
 * from constant memory.  This implementation includes the 
 * SM-aware optimizations of corrSharedSMSums.cuh and also 
 * unrolls the innermost loop 4x.
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

extern __shared__ unsigned char LocalBlock[];


template<bool bSM1>
__global__ void 
corrShared4_kernel( 
    float *pCorr, size_t CorrPitch, 
    int wTile,
    int wTemplate, int hTemplate,
    float xOffset, float yOffset,
    float cPixels, float fDenomExp, int SharedPitch,
    float xUL, float yUL, int w, int h )
{ 
    int uTile = blockIdx.x*wTile;
    int vTile = blockIdx.y*blockDim.y;
    int v = vTile + threadIdx.y;

    float *pOut = (float *) (((char *) pCorr)+v*CorrPitch);

    for ( int row = threadIdx.y; 
              row < blockDim.y+hTemplate; 
              row += blockDim.y ) {
        int SharedIdx = row * SharedPitch;
        for ( int col = threadIdx.x; 
                  col < wTile+wTemplate; 
                  col += blockDim.x ) {

            LocalBlock[SharedIdx+col] = 
                tex2D( texImage, 
                      (float) (uTile+col+xUL+xOffset), 
                      (float) (vTile+row+yUL+yOffset) );

        }
    }

    __syncthreads();

    for ( int col = threadIdx.x; 
              col < wTile; 
              col += blockDim.x ) {

        int SumI = 0;
        int SumISq = 0;
        int SumIT = 0;
        int idx = 0;
        int SharedIdx = threadIdx.y * SharedPitch + col;
        for ( int j = 0; j < hTemplate; j++ ) {
            for ( int i = 0; i < wTemplate/4; i++) { 
                corrSharedAccumulate<bSM1>( SumI, SumISq, SumIT, LocalBlock[SharedIdx+i*4+0], g_Tpix[idx++] );
                corrSharedAccumulate<bSM1>( SumI, SumISq, SumIT, LocalBlock[SharedIdx+i*4+1], g_Tpix[idx++] );
                corrSharedAccumulate<bSM1>( SumI, SumISq, SumIT, LocalBlock[SharedIdx+i*4+2], g_Tpix[idx++] );
                corrSharedAccumulate<bSM1>( SumI, SumISq, SumIT, LocalBlock[SharedIdx+i*4+3], g_Tpix[idx++] );
            }
            SharedIdx += SharedPitch;
        }
        if ( uTile+col < w && v < h ) {
            pOut[uTile+col] = CorrelationValue( SumI, SumISq, SumIT, g_SumT, cPixels, fDenomExp );
        }
    }
    __syncthreads();
}


void
corrShared4( 
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
    int device;
    cudaDeviceProp props;
    cudaError_t status;

    cuda(GetDevice( &device ) );
    cuda(GetDeviceProperties( &props, device ) );
    if ( sharedMem > props.sharedMemPerBlock ) {
        dim3 tcThreads(32, 16, 1);
        dim3 tcBlocks;
        tcBlocks.x = INTCEIL(w,threads.x); 
        tcBlocks.y = INTCEIL(h,threads.y); 
        tcBlocks.z = 1;
        return corrTexConstant( 
            dCorr, CorrPitch,
            wTile,
            wTemplate, hTemplate,
            cPixels,
            fDenomExp,
            sharedPitch,
            xOffset, yOffset,
            xTemplate, yTemplate,
            xUL, yUL, w, h,
            tcThreads, tcBlocks,
            sharedMem );
    }
    if ( wTemplate % 4 ) {
        if ( props.major == 1 ) {
            corrSharedSM_kernel<true><<<blocks, threads, sharedMem>>>(
                dCorr, CorrPitch,
                wTile,
                wTemplate, hTemplate,
                (float) xOffset, (float) yOffset,
                cPixels, fDenomExp, 
                sharedPitch, 
                (float) xUL, (float) yUL, w, h );
        }
        else {
            corrSharedSM_kernel<false><<<blocks, threads, sharedMem>>>(
                dCorr, CorrPitch,
                wTile,
                wTemplate, hTemplate,
                (float) xOffset, (float) yOffset,
                cPixels, fDenomExp, 
                sharedPitch, 
                (float) xUL, (float) yUL, w, h );
        }
    }
    if ( props.major == 1 ) {
        corrShared4_kernel<true><<<blocks, threads, sharedMem>>>(
            dCorr, CorrPitch,
            wTile,
            wTemplate, hTemplate,
            (float) xOffset, (float) yOffset,
            cPixels, fDenomExp, 
            sharedPitch, 
            (float) xUL, (float) yUL, w, h );
    }
    else {
        corrShared4_kernel<false><<<blocks, threads, sharedMem>>>(
            dCorr, CorrPitch,
            wTile,
            wTemplate, hTemplate,
            (float) xOffset, (float) yOffset,
            cPixels, fDenomExp, 
            sharedPitch, 
            (float) xUL, (float) yUL, w, h );
    }
Error:
    return;
}
