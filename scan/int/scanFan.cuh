/*
 *
 * scanFan.cuh
 *
 * Scan-then-fan formulation of scan algorithm.
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

#include <assert.h>

template<class T, bool bWriteSpine>
__global__ void
scanAndWritePartials( 
    T *out, 
    T *gPartials, 
    const T *in, 
    size_t N, 
    size_t numBlocks )
{
    extern volatile __shared__ T sPartials[];
    const int tid = threadIdx.x;
    volatile T *myShared = sPartials+tid;

    for ( size_t iBlock = blockIdx.x; 
                 iBlock < numBlocks; 
                 iBlock += gridDim.x ) {
        size_t index = iBlock*blockDim.x+tid;

        *myShared = (index < N) ? in[index] : 0;
        __syncthreads();

        T sum = scanBlock( myShared );
        __syncthreads();
        if ( index < N ) {
            out[index] = *myShared;
        }
        //
        // write the spine value to global memory
        //
        if ( bWriteSpine && (threadIdx.x==(blockDim.x-1)) )
        {
            gPartials[iBlock] = sum;
        }
    }
}

template<class T>
__global__ void
scanAddBaseSums( 
    T *out, 
    T *gBaseSums, 
    size_t N, 
    size_t numBlocks )
{
    const int tid = threadIdx.x;

    T fan_value = 0;
    for ( size_t iBlock = blockIdx.x; 
                 iBlock < numBlocks; 
                 iBlock += gridDim.x ) {
        size_t index = iBlock*blockDim.x+tid;
        if ( iBlock > 0 ) {
            fan_value = gBaseSums[iBlock-1];
        }
        out[index] += fan_value;
    }
}

/*
 * scanFan - Scan subarrays of length b, writing partial sums
 *     to device memory; scan the partial sums; then add
 *     the partial sum corresponding to each subarray to
 *     all elements in the subarray.
 * The base scan algorithm (scanAndWritePartials) can only scan
 *     b elements, so if more than b partial sums are
 *     needed, this routine recurses.
 *
 * This routine performs an inclusive scan.
 *
 */
template<class T>
void
scanFan( T *out, const T *in, size_t N, int b )
{
    cudaError_t status;

    if ( N <= b ) {
        scanAndWritePartials<T, false><<<1,b,b*sizeof(T)>>>( 
            out, 0, in, N, 1 );
        return;
    }

    //
    // device pointer to array of partial sums in global memory
    //
    T *gPartials = 0;

    //
    // ceil(N/b)
    //
    size_t numPartials = (N+b-1)/b;

    //
    // number of CUDA threadblocks to use.  The kernels are 
    // blocking agnostic, so we can clamp to any number 
    // within CUDA's limits and the code will work.
    //
    const unsigned int maxBlocks = 150;   // maximum blocks to launch
    unsigned int numBlocks = min( numPartials, maxBlocks );

    cuda(Malloc( &gPartials, 
                              numPartials*sizeof(T) ) );

    scanAndWritePartials<T, true><<<numBlocks,b,b*sizeof(T)>>>( 
        out, gPartials, in, N, numPartials );
    scanFan<T>( gPartials, gPartials, numPartials, b );
    scanAddBaseSums<T><<<numBlocks, b>>>( out, gPartials, N, numPartials );

 Error:
    cudaFree( gPartials );
}
