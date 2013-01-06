/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

#include <assert.h>

#include "reduceBlock.cuh"

template<class T>
__host__ __device__ bool
isOdd( T x )
{
    return x & 1;
}

template<class T, int numThreads>
__device__ void
predicateReduceSubarray_odd( 
    int *gPartials, 
    const T *in, 
    size_t iBlock, 
    size_t N, 
    int elementsPerPartial )
{
    extern volatile __shared__ int sPartials[];
    const int tid = threadIdx.x;

    size_t baseIndex = iBlock*elementsPerPartial;

    int sum = 0;
    for ( int i = tid; i < elementsPerPartial; i += blockDim.x ) {
        size_t index = baseIndex+i;
        if ( index < N )
            sum += isOdd( in[index] );
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
predicateReduceSubarrays_odd( 
    int *gPartials, 
    const T *in, 
    size_t N, 
    int elementsPerPartial )
{
    extern volatile __shared__ int sPartials[];

    for ( int iBlock = blockIdx.x; 
          iBlock*elementsPerPartial < N; 
          iBlock += gridDim.x )
    {
        predicateReduceSubarray_odd<T,numThreads>( 
            gPartials, 
            in, 
            iBlock, 
            N, 
            elementsPerPartial );
    }
}

template<class T>
void
predicateReduceSubarrays_odd( int *gPartials, const T *in, size_t N, int numPartials, int cBlocks, int cThreads )
{
    switch ( cThreads ) {
        case 128: return predicateReduceSubarrays_odd<T,128><<<cBlocks, 128, 128*sizeof(T)>>>( gPartials, in, N, numPartials );
        case 256: return predicateReduceSubarrays_odd<T,256><<<cBlocks, 256, 256*sizeof(T)>>>( gPartials, in, N, numPartials );
        case 512: return predicateReduceSubarrays_odd<T,512><<<cBlocks, 512, 512*sizeof(T)>>>( gPartials, in, N, numPartials );
        case 1024: return predicateReduceSubarrays_odd<T,1024><<<cBlocks, 1024, 1024*sizeof(T)>>>( gPartials, in, N, numPartials );
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
streamCompact_odd_kernel( 
    T *out, 
    int *outCount,
    const int *gBaseSums, 
    const T *in, 
    size_t N, 
    size_t elementsPerPartial )
{
    extern volatile __shared__ int sPartials[];
    const int tid = threadIdx.x;
    int sIndex = scanSharedIndex<bZeroPad>( threadIdx.x );

    if ( bZeroPad ) {
        sPartials[sIndex-16] = 0;
    }
    // exclusive scan element gBaseSums[blockIdx.x]
    int base_sum = 0;
    if ( blockIdx.x && gBaseSums ) {
        base_sum = gBaseSums[blockIdx.x-1];
    }
    for ( size_t i = 0;
                 i < elementsPerPartial;
                 i += blockDim.x ) {
        size_t index = blockIdx.x*elementsPerPartial + i + tid;
        int value = (index < N) ? in[index] : 0;
        sPartials[sIndex] = (index < N) ? isOdd( value ) : 0;
        __syncthreads();

        scanBlock<int,bZeroPad>( sPartials+sIndex );
        __syncthreads();
        if ( index < N && isOdd( value ) ) {
            int outIndex = base_sum;
            if ( tid ) {
                int index = scanSharedIndex<bZeroPad>(tid-1);
                outIndex += sPartials[index];
            }
            out[outIndex] = value;
        }
        __syncthreads();

        // carry forward from this block to the next.
        {
            int index = scanSharedIndex<bZeroPad>( blockDim.x-1 );
            base_sum += sPartials[ index ];
        }
        __syncthreads();
    }
    if ( threadIdx.x == 0 && blockIdx.x == 0 ) {
        if ( gBaseSums ) {
            *outCount = gBaseSums[gridDim.x-1];
        }
        else {
            int index = scanSharedIndex<bZeroPad>( blockDim.x-1 );
            *outCount = sPartials[ index ];
        }
    }
}



/*
 * streamCompact_odd
 *
 *     This sample illustrates how to scan predicates,
 *     with an example predicate of testing integers
 *     and emitting values that are odd.
 *
 * The algorithm is implemented using the 2-pass scan 
 *     algorithm, counting the true predicates with a
 *     reduction pass; scanning the predicates; then 
 *     passing over the data again, evaluating the 
 *     predicates again and using the scanned predicate
 *     values as indices to write the output for which
 *     the predicate is true.
 */

#define MAX_PARTIALS 300

__device__ int g_globalPartials[MAX_PARTIALS];

template<class T, bool bZeroPad>
void
streamCompact_odd( T *out, int *outCount, const T *in, size_t N, int b )
{
    int sBytes = scanSharedMemory<int,bZeroPad>( b );

    if ( N <= b ) {
        return streamCompact_odd_kernel<T, bZeroPad><<<1,b,sBytes>>>( 
            out, outCount, 0, in, N, N );
    }

    cudaError_t status;
    int *gPartials = 0;
    status = cudaGetSymbolAddress( 
        (void **) &gPartials, 
        g_globalPartials );

    if ( cudaSuccess ==  status ) {
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

        predicateReduceSubarrays_odd<T>( 
            gPartials, 
            in, 
            N, 
            elementsPerPartial, 
            numBlocks, 
            b );
        predicateScan_kernel<int, bZeroPad><<<1,b,sBytes>>>( 
            gPartials, 
            gPartials, 
            numPartials, 
            numPartials);
        streamCompact_odd_kernel<T, bZeroPad><<<numBlocks,b,sBytes>>>(
            out, 
            outCount,
            gPartials, 
            in, 
            N, 
            elementsPerPartial );
    }
}
