/*
 *
 * reductionVectorized.cuh
 *
 * Bandwidth-optimized reduction. Two changes take the reduction from
 * "correct" to memory-bound:
 *
 *   1. Vectorized loads. Each thread reads four ints at a time as an int4,
 *      so the load path issues 128-bit transactions and needs fewer
 *      instructions to cover the same bytes.
 *   2. Single pass. A grid-wide result is accumulated with one global
 *      atomicAdd per block, so only one kernel launch is needed and no
 *      intermediate array is staged through global memory.
 *
 * The block reduction reuses blockReduceCG() (cooperative_groups::reduce).
 * *out must be initialized to 0 before launch.
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

#ifndef __REDUCTION_VECTORIZED_CUH__
#define __REDUCTION_VECTORIZED_CUH__

#include "reductionWarpShuffleCG.cuh"

__global__ void
ReductionVector_kernel( int *out, const int *in, size_t N )
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>( block );
    extern __shared__ int sPartials[];

    int sum = 0;

    // Body: consume the input four ints at a time with 128-bit loads.
    const size_t N4 = N / 4;
    const int4 *in4 = reinterpret_cast<const int4 *>( in );
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
          i < N4;
          i += blockDim.x*gridDim.x ) {
        int4 v = in4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // Tail: the up-to-three elements past the last full int4.
    for ( size_t i = 4*N4 + blockIdx.x*blockDim.x + threadIdx.x;
          i < N;
          i += blockDim.x*gridDim.x ) {
        sum += in[i];
    }

    sum = blockReduceCG<int>( block, warp, sum, sPartials );
    if ( threadIdx.x == 0 )
        atomicAdd( out, sum );
}

void
ReductionVector( int *answer, int *partial,
                 const int *in, size_t N,
                 int numBlocks, int numThreads )
{
    int sharedBytes = (numThreads/32) * sizeof(int);
    cudaMemset( answer, 0, sizeof(int) );
    ReductionVector_kernel<<< numBlocks, numThreads, sharedBytes >>>(
        answer, in, N );
}

#endif // __REDUCTION_VECTORIZED_CUH__
