/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

#include <assert.h>

#include "reduceBlock.cuh"

template<class T, int numThreads>
__device__ void
predicateReduceSubarray_floatLT( 
    int *gPartials, 
    const T *in, 
    size_t iBlock, 
    size_t N, 
    int elementsPerPartial,
    float floatLT )
{
    extern volatile __shared__ int sPartials[];
    const int tid = threadIdx.x;

    size_t baseIndex = iBlock*elementsPerPartial;

    int sum = 0;
    for ( int i = tid; i < elementsPerPartial; i += blockDim.x ) {
        size_t index = baseIndex+i;
        if ( index < N )
            sum += in[index] < floatLT;
    }
    sPartials[tid] = sum;
    __syncthreads();

    reduceBlock<int,numThreads>( &gPartials[iBlock], sPartials );
}

/*
 * Compute the reductions of each subarray of size
 * elementsPerPartial, and write them to gPartials.
 */
template<class T, int numThreads>
__global__ void
predicateReduceSubarrays_floatLT( 
    int *gPartials, 
    const T *in, 
    size_t N, 
    int elementsPerPartial,
    float floatLT )
{
    extern volatile __shared__ int sPartials[];

    for ( int iBlock = blockIdx.x; 
          iBlock*elementsPerPartial < N; 
          iBlock += gridDim.x )
    {
        predicateReduceSubarray_floatLT<T,numThreads>( 
            gPartials, 
            in, 
            iBlock, 
            N, 
            elementsPerPartial,
            floatLT );
    }
}

template<class T>
void
predicateReduceSubarrays_floatLT( int *gPartials, const T *in, size_t N, int numPartials, float floatLT, int cBlocks, int cThreads )
{
    switch ( cThreads ) {
        case 128: return predicateReduceSubarrays_floatLT<T,128><<<cBlocks, 128, 128*sizeof(T)>>>( gPartials, in, N, numPartials, floatLT );
        case 256: return predicateReduceSubarrays_floatLT<T,256><<<cBlocks, 256, 256*sizeof(T)>>>( gPartials, in, N, numPartials, floatLT );
        case 512: return predicateReduceSubarrays_floatLT<T,512><<<cBlocks, 512, 512*sizeof(T)>>>( gPartials, in, N, numPartials, floatLT );
    }
}

template<class T, bool bZeroPad>
__global__ void
predicateScan_kernel( 
    T *out, 
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
    for ( size_t i = 0;
                 i < elementsPerPartial;
                 i += blockDim.x )
    {
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
        base_sum += sPartials[ scanSharedIndex<bZeroPad>( blockDim.x-1 ) ];
        __syncthreads();
    }
}

template<class T, bool bZeroPad>
__global__ void
streamCompact_floatLT( 
    T *out, 
    int *outLT,
    const int *gBaseSums, 
    const T *in, 
    size_t N, 
    size_t elementsPerPartial,
    float floatLT )
{
    extern volatile __shared__ int sPartials[];
    const int tid = threadIdx.x;
    int sIndex = scanSharedIndex<bZeroPad>( threadIdx.x );

    if ( bZeroPad ) {
        sPartials[sIndex-16] = 0;
    }
    int base_sum = 0;
    if ( blockIdx.x && gBaseSums ) {
        base_sum = gBaseSums[blockIdx.x-1];
    }
    for ( size_t i = 0;
                 i < elementsPerPartial;
                 i += blockDim.x ) {
        size_t index = blockIdx.x*elementsPerPartial + i + tid;
        float value = (index < N) ? in[index] : 0.0f;
        sPartials[sIndex] = (index < N) ? value<floatLT : 0;
        __syncthreads();

        scanBlock<int,bZeroPad>( sPartials+sIndex );
        __syncthreads();
        if ( index < N && value < floatLT ) {
            int outIndex = base_sum;
            if ( tid ) {
                outIndex += sPartials[scanSharedIndex<bZeroPad>(tid-1)];
            }
            out[outIndex] = value;
        }
        __syncthreads();

        // carry forward from this block to the next.
        base_sum += sPartials[ scanSharedIndex<bZeroPad>( blockDim.x-1 ) ];
        __syncthreads();
    }
    if ( threadIdx.x == 0 && blockIdx.x == 0 ) {
        if ( gBaseSums ) {
            *outLT = gBaseSums[gridDim.x-1];
        }
        else {
            *outLT = sPartials[ scanSharedIndex<bZeroPad>( blockDim.x-1 ) ];
        }
    }
}



/*
 * streamCompact_floatLT
 *     Takes a float parameter and a float array
 *     and writes to the output all floats that are 
 *     less than the input float.
 *
 *     This sample illustrates how to scan predicates,
 *     with an example predicate of comparing FP values
 *     and emitting values below a threshold.
 *
 * The algorithm is implemented using the 2-pass scan 
 *     algorithm, counting the true predicates; scanning
 *     the predicates; then passing over the data again, 
 *     evaluating the predicates again and using the 
 *     scanned predicate values as indices to write the
 *     output for which the predicate is true.
 */

#define MAX_PARTIALS 300

__device__ int g_globalPartials[MAX_PARTIALS];

template<class T, bool bZeroPad>
void
streamCompact_floatLT( T *out, int *outLT, const T *in, size_t N, int b, float floatLT )
{
    int sBytes = scanSharedMemory<int,bZeroPad>( b );

    if ( N <= b ) {
        return streamCompact_floatLT<T, bZeroPad><<<1,b,sBytes>>>( 
            out, outLT, 0, in, N, N, floatLT );
    }

    cudaError_t status;
    int *gPartials = 0;
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

        predicateReduceSubarrays_floatLT<T>( 
            gPartials, 
            in, 
            N, 
            elementsPerPartial, 
            floatLT,
            numBlocks, 
            b );
        predicateScan_kernel<int, bZeroPad><<<1,b,sBytes>>>( 
            gPartials, 
            gPartials, 
            numPartials, 
            numPartials);
        streamCompact_floatLT<T, bZeroPad><<<numBlocks,b,sBytes>>>(
            out, 
            outLT,
            gPartials, 
            in, 
            N, 
            elementsPerPartial,
            floatLT );
    }
}
