/*
 *
 * reductionWarpShuffleCG.cuh
 *
 * Block reduction built on cooperative_groups::reduce -- the modern idiom.
 * A warp reduces with cg::reduce (which the compiler lowers to the
 * __shfl_xor_sync butterfly); one partial per warp is staged through shared
 * memory; then the first warp reduces those partials. No power-of-2 block
 * size is required, and there is no hand-written log-step loop.
 *
 * Two-pass driver, like reductionWarpShuffle.cuh: the same kernel is invoked
 * once over the input and once over the per-block partial sums.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
 *
 */

#ifndef __REDUCTION_WARP_SHUFFLE_CG_CUH__
#define __REDUCTION_WARP_SHUFFLE_CG_CUH__

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

//
// Reduce v across the whole thread block. The result is returned to
// thread 0; other threads' return values are undefined. sPartials must
// have room for one T per warp (blockDim.x/32 elements).
//
template<class T>
__device__ T
blockReduceCG( cg::thread_block block,
               cg::thread_block_tile<32> warp,
               T v, T *sPartials )
{
    v = cg::reduce( warp, v, cg::plus<T>() );   // one value per warp
    if ( warp.thread_rank() == 0 )
        sPartials[warp.meta_group_rank()] = v;
    block.sync();

    if ( warp.meta_group_rank() == 0 ) {
        v = ( warp.thread_rank() < warp.meta_group_size() ) ?
                sPartials[warp.thread_rank()] : (T) 0;
        v = cg::reduce( warp, v, cg::plus<T>() );
    }
    return v;
}

__global__ void
ReductionCG_kernel( int *out, const int *in, size_t N )
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>( block );
    extern __shared__ int sPartials[];

    int sum = 0;
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
          i < N;
          i += blockDim.x*gridDim.x ) {
        sum += in[i];
    }

    sum = blockReduceCG<int>( block, warp, sum, sPartials );
    if ( threadIdx.x == 0 )
        out[blockIdx.x] = sum;
}

void
ReductionCG( int *answer, int *partial,
             const int *in, size_t N,
             int numBlocks, int numThreads )
{
    int sharedBytes = (numThreads/32) * sizeof(int);
    ReductionCG_kernel<<< numBlocks, numThreads, sharedBytes >>>(
        partial, in, N );
    ReductionCG_kernel<<< 1, numThreads, sharedBytes >>>(
        answer, partial, numBlocks );
}

#endif // __REDUCTION_WARP_SHUFFLE_CG_CUH__
