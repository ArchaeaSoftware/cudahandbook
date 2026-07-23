/*
 *
 * corrShared4Dp4a.cuh
 *
 * Prototype variant of corrShared4 that replaces the 4x-unrolled scalar
 * inner loop with three DP4A (__dp4a) instructions per group of four
 * template pixels.  See corrShared4Dp4aSums.cuh for the detailed notes on
 * packing, endianness, and the unsigned-__dp4a requirement.
 *
 * Requires SM 6.1+ for __dp4a.  This is the coefficient-only variant used
 * for timing; the Sums variant validates correctness.
 *
 * Copyright (c) 2012-2026, Archaea Software, LLC.
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

extern __shared__ unsigned char LocalBlock[];


__global__ void
corrShared4Dp4a_kernel(
    float *pCorr, size_t CorrPitch,
    cudaTextureObject_t texImage,
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
                tex2D<unsigned char>( texImage,
                      (float) (uTile+col+xUL+xOffset),
                      (float) (vTile+row+yUL+yOffset) );

        }
    }

    __syncthreads();

    for ( int col = threadIdx.x;
              col < wTile;
              col += blockDim.x ) {

        unsigned SumI = 0;
        unsigned SumISq = 0;
        unsigned SumIT = 0;
        int idx = 0;

        // Unaligned image window; loop-invariant realignment (see the Sums
        // variant's comments for the full derivation).
        int byteOff = col & 3;
        int shift = byteOff * 8;
        int alignedIdx = threadIdx.y * SharedPitch + (col & ~3);

        for ( int j = 0; j < hTemplate; j++ ) {
            const unsigned *pRow = (const unsigned *)( LocalBlock + alignedIdx );
            for ( int i = 0; i < wTemplate/4; i++ ) {
                unsigned w0 = pRow[i];
                unsigned w1 = pRow[i+1];
                unsigned I4 = __funnelshift_r( w0, w1, shift );
                unsigned T4 = *(const unsigned *)( g_Tpix + idx ); idx += 4;
                SumI   = __dp4a( I4, 0x01010101u, SumI );
                SumISq = __dp4a( I4, I4,          SumISq );
                SumIT  = __dp4a( I4, T4,          SumIT );
            }
            alignedIdx += SharedPitch;
        }
        if ( uTile+col < w && v < h ) {
            pOut[uTile+col] = CorrelationValue( SumI, SumISq, SumIT, g_SumT, cPixels, fDenomExp );
        }
    }
    __syncthreads();
}


void
corrShared4Dp4a(
    float *dCorr, int CorrPitch,
    cudaTextureObject_t texImage, cudaTextureObject_t texTemplate,
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
            texImage, texTemplate,
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
        corrSharedSM_kernel<<<blocks, threads, sharedMem>>>(
            dCorr, CorrPitch,
            texImage,
            wTile,
            wTemplate, hTemplate,
            (float) xOffset, (float) yOffset,
            cPixels, fDenomExp,
            sharedPitch,
            (float) xUL, (float) yUL, w, h );
    }
    corrShared4Dp4a_kernel<<<blocks, threads, sharedMem>>>(
        dCorr, CorrPitch,
        texImage,
        wTile,
        wTemplate, hTemplate,
        (float) xOffset, (float) yOffset,
        cPixels, fDenomExp,
        sharedPitch,
        (float) xUL, (float) yUL, w, h );
Error:
    return;
}
