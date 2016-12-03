/*
 *
 * scanReduceThenScan.cuh
 *
 * Reduce-then-scan formulation of scan.
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

#include "reduceBlock.cuh"

template<class T, int numThreads>
__global__ void
scanReduceBlocks( T *gPartials, const T *in, size_t N )
{
    extern volatile __shared__ T sPartials[];

    const int tid = threadIdx.x;
    gPartials += blockIdx.x;
    for ( size_t i = blockIdx.x*blockDim.x; 
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        size_t index = i+tid;
        sPartials[tid] = (index < N) ? in[index] : 0;
        __syncthreads();

        reduceBlock<T,numThreads>( gPartials, sPartials );
        __syncthreads();
        gPartials += gridDim.x;
    }
}

template<class T>
void
scanReduceBlocks( 
    T *gPartials, 
    const T *in, 
    size_t N, 
    int numThreads, 
    int numBlocks )
{
    switch ( numThreads ) {
        case  128: return scanReduceBlocks<T, 128><<<numBlocks,  128,  128*sizeof(T)>>>( gPartials, in, N );
        case  256: return scanReduceBlocks<T, 256><<<numBlocks,  256,  256*sizeof(T)>>>( gPartials, in, N );
        case  512: return scanReduceBlocks<T, 512><<<numBlocks,  512,  512*sizeof(T)>>>( gPartials, in, N );
        case 1024: return scanReduceBlocks<T,1024><<<numBlocks, 1024, 1024*sizeof(T)>>>( gPartials, in, N );
    }
}

template<class T>
__global__ void
scanWithBaseSums( 
    T *out, 
    const T *gBaseSums, 
    const T *in, 
    size_t N, 
    size_t numBlocks )
{
    extern volatile __shared__ T sPartials[];
    const int tid = threadIdx.x;

    for ( size_t iBlock = blockIdx.x; 
                 iBlock < numBlocks; 
                 iBlock += gridDim.x ) {
        T base_sum = 0;
        size_t index = iBlock*blockDim.x+tid;

        if ( iBlock > 0 && gBaseSums ) {
            base_sum = gBaseSums[iBlock-1];
        }
        sPartials[tid] = (index < N) ? in[index] : 0;
        __syncthreads();

        scanBlock( sPartials+tid );
        __syncthreads();
        if ( index < N ) {
            out[index] = sPartials[tid]+base_sum;
        }
    }
}

/*
 * scanReduceThenScan - Compute reductions of subarrays of length b, 
 *     writing partial sums; scan the partial sums; then
 *     perform a scan of each subarray, adding its
 *     corresponding partial sum on the way.
 * The base scan algorithm (scanAndWritePartials) can only scan
 *     b elements, so if more than b partial sums are
 *     needed, this routine recurses.
 *
 * b must be a power of 2 that is also a legitimate
 *     CUDA block size: 128, 256, or 512.
 *
 * This routine performs an inclusive scan.
 *
 */
template<class T>
void
scanReduceThenScan( T *out, const T *in, size_t N, int b )
{
    cudaError_t status;

    if ( N <= b ) {
        return scanWithBaseSums<T><<<1,b,b*sizeof(T)>>>( 
            out, 0, in, N, 1 );
    }

    //
    // device pointer to array of partial sums in global memory
    //
    T *gPartials = 0;

    //
    // ceil(N/b) = number of partial sums to compute
    //
    size_t numPartials = (N+b-1)/b;

    //
    // number of CUDA threadblocks to use.  The kernels are blocking
    // agnostic, so we can clamp to any number within CUDA's limits
    // and the code will work.
    //
    const unsigned int maxBlocks = 150;
    unsigned int numBlocks = min( numPartials, maxBlocks );

    cuda(Malloc( &gPartials, numPartials*sizeof(T) ) );

    scanReduceBlocks<T>( gPartials, in, N, b, numBlocks );
    scanReduceThenScan<T>( gPartials, gPartials, numPartials, b );
    scanWithBaseSums<T><<<numBlocks,b,b*sizeof(T)>>>( 
        out, 
        gPartials, 
        in, 
        N, 
        numPartials );
 Error:
    cudaFree( gPartials );

}
