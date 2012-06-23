/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

//
// reads N ints and writes an intermediate sum per block
// numThreads must be a power of 2!
//

template<class ReductionType, class T, unsigned int numThreads, unsigned int numBlocks>
__global__ void
Reduction5_kernel( ReductionType *out, const T *in, size_t N )
{
    SharedMemory<ReductionType> sPartials;
    const unsigned int tid = threadIdx.x;
    ReductionType sum;
    for ( size_t i = blockIdx.x*numThreads + tid;
          i < N;
          i += numThreads*numBlocks/*gridDim.x*/ )
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    if (numThreads >= 512) { 
        if (tid < 256) {
            sPartials[tid] += sPartials[tid + 256];
        } 
        __syncthreads();
    }
    if (numThreads >= 256) {
        if (tid < 128) {
            sPartials[tid] += sPartials[tid + 128];
        } 
        __syncthreads();
    }
    if (numThreads >= 128) {
        if (tid <  64) { 
            sPartials[tid] += sPartials[tid +  64];
        } 
        __syncthreads();
    }
    // warp synchronous at the end
    if ( tid < 32 ) {
        volatile ReductionType *wsSum = sPartials;
        if (numThreads >=  64) { wsSum[tid] += wsSum[tid + 32]; }
        if (numThreads >=  32) { wsSum[tid] += wsSum[tid + 16]; }
        if (numThreads >=  16) { wsSum[tid] += wsSum[tid +  8]; }
        if (numThreads >=   8) { wsSum[tid] += wsSum[tid +  4]; }
        if (numThreads >=   4) { wsSum[tid] += wsSum[tid +  2]; }
        if (numThreads >=   2) { wsSum[tid] += wsSum[tid +  1]; }
        if ( tid == 0 ) {
            out[blockIdx.x] = sPartials[0];
        }
    }
}

template<class ReductionType, class T, unsigned int numThreads>
void
Reduction5_template( ReductionType *answer, ReductionType *partial, const T *in, size_t N, int numBlocks )
{
    Reduction5_kernel<ReductionType, T, numThreads, 120><<< 120/*numBlocks*/, numThreads, numThreads*sizeof(ReductionType)>>>( partial, in, N );
    Reduction5_kernel<ReductionType, ReductionType, numThreads, 1><<<         1, numThreads, numThreads*sizeof(ReductionType)>>>( answer, partial, numBlocks );
}

template<class ReductionType, class T>
void
Reduction5( ReductionType *out, ReductionType *partial, const T *in, size_t N, int numBlocks, int numThreads )
{
    if ( N < numBlocks*numThreads ) {
        numBlocks = (N+numThreads-1)/numThreads;
    }
    switch ( numThreads ) {
        case   1: return Reduction5_template<ReductionType, T,  1>( out, partial, in, N, numBlocks );
        case   2: return Reduction5_template<ReductionType, T,  2>( out, partial, in, N, numBlocks );
        case   4: return Reduction5_template<ReductionType, T,  4>( out, partial, in, N, numBlocks );
        case   8: return Reduction5_template<ReductionType, T,  8>( out, partial, in, N, numBlocks );
        case  16: return Reduction5_template<ReductionType, T, 16>( out, partial, in, N, numBlocks );
        case  32: return Reduction5_template<ReductionType, T, 32>( out, partial, in, N, numBlocks );
        case  64: return Reduction5_template<ReductionType, T, 64>( out, partial, in, N, numBlocks );
        case 128: return Reduction5_template<ReductionType, T,128>( out, partial, in, N, numBlocks );
        case 256: return Reduction5_template<ReductionType, T,256>( out, partial, in, N, numBlocks );
        case 512: return Reduction5_template<ReductionType, T,512>( out, partial, in, N, numBlocks );
    }
}
