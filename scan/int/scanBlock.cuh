/*
 *
 * scanBlock.cuh
 *
 * Utilities to perform scan for a threadblock.
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

#ifndef __SCAN_BLOCK_CUH__
#define __SCAN_BLOCK_CUH__

/*
 * scanBlock - scans block in shared memory
 *    sPartials is the shared memory pointer to
 *    this thread's input/output element.
 *
 * warpPartials[] is all of shared memory -
 *    the partial sums from each warp are
 *    written to warpPartials, scanned,
 *    and used as base sums to compute the
 *    output.
 *
 * If threadIdx.x==blockDim.x-1, this function 
 *    returns the reduction of all elements in
 *    shared memory.
 *
 */
template<class T, bool bZeroPadded>
inline __device__ T
scanBlock( volatile T *sPartials )
{
    extern __shared__ T warpPartials[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    //
    // Compute this thread's partial sum
    //
    T sum = scanWarp<T,bZeroPadded>( sPartials );
    __syncthreads();

    //
    // Write each warp's reduction to shared memory
    // 
    if ( lane == 31 ) {
        warpPartials[16+warpid] = sum;
    }
    __syncthreads();

    //
    // Have one warp scan reductions
    //
    if ( warpid==0 ) {
        scanWarp<T,bZeroPadded>( 16+warpPartials+tid );
    }
    __syncthreads();

    //
    // Fan out the exclusive scan element (obtained
    // by the conditional and the decrement by 1)
    // to this warp's pending output
    //
    if ( warpid > 0 ) {
        sum += warpPartials[16+warpid-1];
    }
    __syncthreads();

    //
    // Write this thread's scan output
    //
    *sPartials = sum;
    __syncthreads();

    //
    // The return value will only be used by caller if it
    // contains the spine value (i.e. the reduction
    // of the array we just scanned).
    //
    return sum;
}

template<class T>
inline __device__ T
scanBlock( volatile T *sPartials )
{
    return scanBlock<T,false>( sPartials );
}

#endif // __SCAN_BLOCK_CUH__
