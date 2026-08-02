/*
 *
 * corrShared4Dp4aSums.cuh
 *
 * Prototype variant of corrShared4Sums that replaces the 4x-unrolled
 * scalar inner loop with three DP4A (__dp4a) instructions per group of
 * four template pixels.  The four image pixels are packed into a single
 * 32-bit register and the sums of I, I*I and I*T are each computed with
 * one 8-bit-dot-product-accumulate.
 *
 * Requires SM 6.1+ for __dp4a (build with e.g. --gpu-architecture sm_61
 * or newer).  Pixels are unsigned char, so the *unsigned* __dp4a overload
 * is mandatory -- the signed overload would treat pixels 128..255 as
 * negative and corrupt SumISq and SumIT.
 *
 * This is the "Sums" variant: it writes SumI/SumISq/SumIT so the test
 * harness can compare them element-wise against the CPU reference.  That
 * comparison is what validates the endianness-sensitive SumIT term, which
 * (unlike SumI and SumISq) depends on image and template bytes landing in
 * matching DP4A lanes.
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
corrShared4Dp4aSums_kernel(
    float *pCorr, size_t CorrPitch,
    int *pI, int *pISq, int *pIT,
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
    pI = (int *) (((char *) pI)+v*CorrPitch);
    pISq = (int *) (((char *) pISq)+v*CorrPitch);
    pIT = (int *) (((char *) pIT)+v*CorrPitch);

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

        // This thread's 4-pixel image windows start at byte offset (col & 3),
        // which is unaligned in shared memory.  SharedPitch is a multiple of
        // 64, so that offset -- and hence the funnel-shift amount -- is the
        // same on every row and can be hoisted out of the loop.
        int byteOff = col & 3;
        int shift = byteOff * 8;
        int alignedIdx = threadIdx.y * SharedPitch + (col & ~3);

        for ( int j = 0; j < hTemplate; j++ ) {
            const unsigned *pRow = (const unsigned *)( LocalBlock + alignedIdx );
            for ( int i = 0; i < wTemplate/4; i++ ) {
                // Realign the four image pixels [SharedIdx+i*4 .. +3] from the
                // two aligned shared words that straddle them.  On the (always
                // little-endian) GPU, funnelshift_r(w0,w1,shift) delivers the
                // four bytes starting at byte (col&3) of w0, packed low-to-high.
                // (A single uint2 load can't replace the pair -- shared LDS.64
                // needs 8-byte alignment and this base is only 4-byte aligned.)
                unsigned w0 = pRow[i];
                unsigned w1 = pRow[i+1];
                unsigned I4 = __funnelshift_r( w0, w1, shift );
                // wTemplate % 4 == 0, so the template word load is aligned and
                // packs its four bytes in the SAME low-to-high order as I4 --
                // that lane agreement is what makes the SumIT dot product valid.
                unsigned T4 = *(const unsigned *)( g_Tpix + idx ); idx += 4;
                SumI   = __dp4a( I4, 0x01010101u, SumI );   // sum of pixels
                SumISq = __dp4a( I4, I4,          SumISq ); // sum of squares
                SumIT  = __dp4a( I4, T4,          SumIT );  // sum of I*T
            }
            alignedIdx += SharedPitch;
        }
        if ( uTile+col < w && v < h ) {
            pI[uTile+col] = SumI;
            pISq[uTile+col] = SumISq;
            pIT[uTile+col] = SumIT;
            pOut[uTile+col] = CorrelationValue( SumI, SumISq, SumIT, g_SumT, cPixels, g_fDenomExp );
        }
    }
    __syncthreads();
}


void
corrShared4Dp4aSums(
    float *dCorr, int CorrPitch,
    int *dSumI, int *dSumISq, int *dSumIT,
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
        return corrTexConstantSums(
            dCorr, CorrPitch,
            dSumI, dSumISq, dSumIT,
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
    if ( wTemplate%4 ) {
        corrSharedSMSums_kernel<<<blocks, threads, sharedMem>>>(
            dCorr, CorrPitch,
            dSumI, dSumISq, dSumIT,
            texImage,
            wTile,
            wTemplate, hTemplate,
            (float) xOffset, (float) yOffset,
            cPixels, fDenomExp,
            sharedPitch,
            (float) xUL, (float) yUL, w, h );
    }
    corrShared4Dp4aSums_kernel<<<blocks, threads, sharedMem>>>(
        dCorr, CorrPitch,
        dSumI, dSumISq, dSumIT,
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
