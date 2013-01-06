/*
 *
 * 2-level formulation of scan - requires exactly three kernel
 * invocations and O(1) extra memory.
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

template<class T, int numThreads>
__device__ void
scanReduceSubarray( 
    T *gPartials, 
    const T *in, 
    size_t iBlock, 
    size_t N, 
    int elementsPerPartial )
{
    extern volatile __shared__ T sPartials[];
    const int tid = threadIdx.x;

    size_t baseIndex = iBlock*elementsPerPartial;

    T sum = 0;
    for ( int i = tid; i < elementsPerPartial; i += blockDim.x ) {
        size_t index = baseIndex+i;
        if ( index < N )
            sum += in[index];
    }
    sPartials[tid] = sum;
    __syncthreads();

    reduceBlock<T,numThreads>( &gPartials[iBlock], sPartials );
}

/*
 * Compute the reductions of each subarray of size
 * elementsPerPartial, and write them to gPartials.
 */
template<class T, int numThreads>
__global__ void
scanReduceSubarrays( 
    T *gPartials, 
    const T *in, 
    size_t N, 
    int elementsPerPartial )
{
    extern volatile __shared__ T sPartials[];

    for ( int iBlock = blockIdx.x; 
          iBlock*elementsPerPartial < N; 
          iBlock += gridDim.x )
    {
        scanReduceSubarray<T,numThreads>( 
            gPartials, 
            in, 
            iBlock, 
            N, 
            elementsPerPartial );
    }
}

template<class T>
void
scanReduceSubarrays( T *gPartials, const T *in, size_t N, int numPartials, int cBlocks, int cThreads )
{
    switch ( cThreads ) {
        case  128: return scanReduceSubarrays<T, 128><<<cBlocks,  128,  128*sizeof(T)>>>( gPartials, in, N, numPartials );
        case  256: return scanReduceSubarrays<T, 256><<<cBlocks,  256,  256*sizeof(T)>>>( gPartials, in, N, numPartials );
        case  512: return scanReduceSubarrays<T, 512><<<cBlocks,  512,  512*sizeof(T)>>>( gPartials, in, N, numPartials );
        case 1024: return scanReduceSubarrays<T,1024><<<cBlocks, 1024, 1024*sizeof(T)>>>( gPartials, in, N, numPartials );
    }
}

template<class T, bool bZeroPad>
__global__ void
scan2Level_kernel( 
    T *out, 
    const T *gBaseSums, 
    const T *in, 
    size_t N, 
    size_t elementsPerPartial )
{
    extern volatile __shared__ T sPartials[];
    const int tid = threadIdx.x;
    int sIndex = scanSharedIndex<bZeroPad>( threadIdx.x );

    if ( bZeroPad ) {
        sPartials[sIndex-16] = 0;
    }
    T base_sum = 0;
    if ( blockIdx.x && gBaseSums ) {
        base_sum = gBaseSums[blockIdx.x-1];
    }
    for ( size_t i = 0;
                 i < elementsPerPartial;
                 i += blockDim.x ) {
        size_t index = blockIdx.x*elementsPerPartial + i + tid;
        sPartials[sIndex] = (index < N) ? in[index] : 0;
        __syncthreads();

        scanBlock<T,bZeroPad>( sPartials+sIndex );
        __syncthreads();
        if ( index < N ) {
            out[index] = sPartials[sIndex]+base_sum;
        }
        __syncthreads();

        // carry forward from this block to the next.
        base_sum += sPartials[ 
            scanSharedIndex<bZeroPad>( blockDim.x-1 ) ];
        __syncthreads();
    }
}

/*
 * scan2Level
 *     Compute reductions of MAX_PARTIALS subarrays, 
 *     writing partial sums; scan the partial sums; 
 *     then perform a scan of each subarray, adding 
 *     its corresponding partial sum on the way.
 *
 * This routine performs an inclusive scan.
 *
 */

#define MAX_PARTIALS 300

__device__ int g_globalPartials[MAX_PARTIALS];

template<class T, bool bZeroPad>
void
scan2Level( T *out, const T *in, size_t N, int b )
{
    int sBytes = scanSharedMemory<T,bZeroPad>( b );

    if ( N <= b ) {
        return scan2Level_kernel<T, bZeroPad><<<1,b,sBytes>>>( 
            out, 0, in, N, N );
    }

    cudaError_t status;
    T *gPartials = 0;
    status = cudaGetSymbolAddress( 
                (void **) &gPartials, 
                g_globalPartials );

    if ( cudaSuccess ==  status )
    {
        //
        // ceil(N/b) = number of partial sums to compute
        //
        size_t numPartials = (N+b-1)/b;

        if ( numPartials > MAX_PARTIALS ) {
            numPartials = MAX_PARTIALS;
        }

        //
        // elementsPerPartial has to be a multiple of b
        // 
        unsigned int elementsPerPartial = (N+numPartials-1)/numPartials;
        elementsPerPartial = b * ((elementsPerPartial+b-1)/b);
        numPartials = (N+elementsPerPartial-1)/elementsPerPartial;

        //
        // number of CUDA threadblocks to use.  The kernels are 
        // blocking agnostic, so we can clamp to any number within 
        // CUDA's limits and the code will work.
        //
        const unsigned int maxBlocks = MAX_PARTIALS;
        unsigned int numBlocks = min( numPartials, maxBlocks );

        scanReduceSubarrays<T>( 
            gPartials, 
            in, 
            N, 
            elementsPerPartial, 
            numBlocks, 
            b );
        scan2Level_kernel<T, bZeroPad><<<1,b,sBytes>>>( 
            gPartials, 
            0, 
            gPartials, 
            numPartials, 
            numPartials );
        scan2Level_kernel<T, bZeroPad><<<numBlocks,b,sBytes>>>(
            out, 
            gPartials, 
            in, 
            N, 
            elementsPerPartial );
    }
}

template<class T>
void
scan2Level_0( T *out, const T *in, size_t N, int b )
{
    scan2Level<T,false>( out, in, N, b );
}

template<class T>
void
scan2Level( T *out, const T *in, size_t N, int b )
{
    scan2Level<T,true>( out, in, N, b );
}
