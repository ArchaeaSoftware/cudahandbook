/*
 *
 * corrShared4Dp4aIlp.cuh
 *
 * ILP experiment on top of the DP4A variant: each thread computes
 * CORR_ILP_NCOL output columns (strided by blockDim.x) with independent
 * accumulator sets, so the kernel has CORR_ILP_NCOL x 3 independent
 * dp4a chains to hide latency at the (already 100%) occupancy.  The
 * template word T4 is loaded once per group and reused across all
 * columns.  Launch with wTile = CORR_ILP_NCOL * blockDim.x.
 *
 * Requires SM 6.1+ for __dp4a.  See corrShared4Dp4aSums.cuh for the
 * packing/endianness notes.
 *
 * Copyright (c) 2012, Archaea Software, LLC.  All rights reserved.
 * (BSD 2-clause; see other corr headers for the full text.)
 *
 */

// 2 columns/thread is the measured sweet spot on sm_86: enough ILP to hide
// the shared-load latency, without the register pressure that costs occupancy
// at 4 or 8 (see the NCOL sweep in the chapter text).
#ifndef CORR_ILP_NCOL
#define CORR_ILP_NCOL 2
#endif

extern __shared__ unsigned char LocalBlock[];

__global__ void
corrShared4Dp4aIlpSums_kernel(
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

    for ( int row = threadIdx.y; row < blockDim.y+hTemplate; row += blockDim.y ) {
        int SharedIdx = row * SharedPitch;
        for ( int col = threadIdx.x; col < wTile+wTemplate; col += blockDim.x ) {
            LocalBlock[SharedIdx+col] =
                tex2D<unsigned char>( texImage,
                       (float) (uTile+col+xUL+xOffset),
                       (float) (vTile+row+yUL+yOffset) );
        }
    }
    __syncthreads();

    unsigned SumI[CORR_ILP_NCOL], SumISq[CORR_ILP_NCOL], SumIT[CORR_ILP_NCOL];
    int shift[CORR_ILP_NCOL], alignBase[CORR_ILP_NCOL];
    #pragma unroll
    for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
        int col = threadIdx.x + c*blockDim.x;
        SumI[c] = SumISq[c] = SumIT[c] = 0;
        shift[c] = (col & 3) * 8;
        alignBase[c] = threadIdx.y * SharedPitch + (col & ~3);
    }

    int idx = 0;
    for ( int j = 0; j < hTemplate; j++ ) {
        for ( int i = 0; i < wTemplate/4; i++ ) {
            int gidx = idx; idx += 4;
#ifdef CORR_ILP_SCALAR
            #pragma unroll
            for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
                int sib = alignBase[c] + (shift[c]>>3) + i*4;   // unaligned byte base
                #pragma unroll
                for ( int k = 0; k < 4; k++ ) {
                    unsigned I = LocalBlock[sib+k];
                    unsigned T = g_Tpix[gidx+k];
                    SumI[c] += I; SumISq[c] += I*I; SumIT[c] += I*T;
                }
            }
#else
            unsigned T4 = *(const unsigned *)( g_Tpix + gidx );  // shared across columns
            #pragma unroll
            for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
                const unsigned *pw = (const unsigned *)( LocalBlock + alignBase[c] + i*4 );
                unsigned I4 = __funnelshift_r( pw[0], pw[1], shift[c] );
                SumI[c]   = __dp4a( I4, 0x01010101u, SumI[c] );
                SumISq[c] = __dp4a( I4, I4,          SumISq[c] );
                SumIT[c]  = __dp4a( I4, T4,          SumIT[c] );
            }
#endif
        }
        #pragma unroll
        for ( int c = 0; c < CORR_ILP_NCOL; c++ ) alignBase[c] += SharedPitch;
    }

    #pragma unroll
    for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
        int col = threadIdx.x + c*blockDim.x;
        if ( uTile+col < w && v < h ) {
            pI[uTile+col] = SumI[c];
            pISq[uTile+col] = SumISq[c];
            pIT[uTile+col] = SumIT[c];
            pOut[uTile+col] = CorrelationValue( SumI[c], SumISq[c], SumIT[c], g_SumT, cPixels, g_fDenomExp );
        }
    }
    __syncthreads();
}

__global__ void
corrShared4Dp4aIlp_kernel(
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

    for ( int row = threadIdx.y; row < blockDim.y+hTemplate; row += blockDim.y ) {
        int SharedIdx = row * SharedPitch;
        for ( int col = threadIdx.x; col < wTile+wTemplate; col += blockDim.x ) {
            LocalBlock[SharedIdx+col] =
                tex2D<unsigned char>( texImage,
                      (float) (uTile+col+xUL+xOffset),
                      (float) (vTile+row+yUL+yOffset) );
        }
    }
    __syncthreads();

    unsigned SumI[CORR_ILP_NCOL], SumISq[CORR_ILP_NCOL], SumIT[CORR_ILP_NCOL];
    int shift[CORR_ILP_NCOL], alignBase[CORR_ILP_NCOL];
    #pragma unroll
    for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
        int col = threadIdx.x + c*blockDim.x;
        SumI[c] = SumISq[c] = SumIT[c] = 0;
        shift[c] = (col & 3) * 8;
        alignBase[c] = threadIdx.y * SharedPitch + (col & ~3);
    }

    int idx = 0;
    for ( int j = 0; j < hTemplate; j++ ) {
        for ( int i = 0; i < wTemplate/4; i++ ) {
            int gidx = idx; idx += 4;
#ifdef CORR_ILP_SCALAR
            #pragma unroll
            for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
                int sib = alignBase[c] + (shift[c]>>3) + i*4;   // unaligned byte base
                #pragma unroll
                for ( int k = 0; k < 4; k++ ) {
                    unsigned I = LocalBlock[sib+k];
                    unsigned T = g_Tpix[gidx+k];
                    SumI[c] += I; SumISq[c] += I*I; SumIT[c] += I*T;
                }
            }
#else
            unsigned T4 = *(const unsigned *)( g_Tpix + gidx );
            #pragma unroll
            for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
                const unsigned *pw = (const unsigned *)( LocalBlock + alignBase[c] + i*4 );
                unsigned I4 = __funnelshift_r( pw[0], pw[1], shift[c] );
                SumI[c]   = __dp4a( I4, 0x01010101u, SumI[c] );
                SumISq[c] = __dp4a( I4, I4,          SumISq[c] );
                SumIT[c]  = __dp4a( I4, T4,          SumIT[c] );
            }
#endif
        }
        #pragma unroll
        for ( int c = 0; c < CORR_ILP_NCOL; c++ ) alignBase[c] += SharedPitch;
    }

    #pragma unroll
    for ( int c = 0; c < CORR_ILP_NCOL; c++ ) {
        int col = threadIdx.x + c*blockDim.x;
        if ( uTile+col < w && v < h ) {
            pOut[uTile+col] = CorrelationValue( SumI[c], SumISq[c], SumIT[c], g_SumT, cPixels, fDenomExp );
        }
    }
    __syncthreads();
}

void
corrShared4Dp4aIlpSums(
    float *dCorr, int CorrPitch,
    int *dSumI, int *dSumISq, int *dSumIT,
    cudaTextureObject_t texImage, cudaTextureObject_t texTemplate,
    int wTile, int wTemplate, int hTemplate,
    float cPixels, float fDenomExp, int sharedPitch,
    int xOffset, int yOffset, int xTemplate, int yTemplate,
    int xUL, int yUL, int w, int h,
    dim3 threads, dim3 blocks, int sharedMem )
{
    corrShared4Dp4aIlpSums_kernel<<<blocks, threads, sharedMem>>>(
        dCorr, CorrPitch, dSumI, dSumISq, dSumIT, texImage,
        wTile, wTemplate, hTemplate,
        (float) xOffset, (float) yOffset, cPixels, fDenomExp,
        sharedPitch, (float) xUL, (float) yUL, w, h );
}

void
corrShared4Dp4aIlp(
    float *dCorr, int CorrPitch,
    cudaTextureObject_t texImage, cudaTextureObject_t texTemplate,
    int wTile, int wTemplate, int hTemplate,
    float cPixels, float fDenomExp, int sharedPitch,
    int xOffset, int yOffset, int xTemplate, int yTemplate,
    int xUL, int yUL, int w, int h,
    dim3 threads, dim3 blocks, int sharedMem )
{
    corrShared4Dp4aIlp_kernel<<<blocks, threads, sharedMem>>>(
        dCorr, CorrPitch, texImage,
        wTile, wTemplate, hTemplate,
        (float) xOffset, (float) yOffset, cPixels, fDenomExp,
        sharedPitch, (float) xUL, (float) yUL, w, h );
}
