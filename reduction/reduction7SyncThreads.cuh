/*
 *
 * reduction7SyncThreads.cuh
 *
 * Header for reduction that uses __syncthreads_count().
 * The function is templated to take a bit count - this is only faster
 * for inputs of a few bits at most.
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

template<const int nBits>
__global__ void
Reduction7_kernel( int *out, const int *in, size_t N )
{
    int blockSum = 0;
    const int tid = threadIdx.x;
    for ( size_t i = blockIdx.x*blockDim.x + tid;
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        int value = in[i];
        for ( int bit = 0; bit < nBits; bit++ ) {
            blockSum += __syncthreads_count( (value&(1<<bit)) ) << bit;
        }
    }

    if ( tid == 0 ) {
        out[blockIdx.x] = blockSum;
    }
}

void
Reduction7( int *answer, int *partial, 
            const int *in, size_t N, 
            int numBlocks, int numThreads )
{
    Reduction7_kernel<32><<< 
        numBlocks, numThreads>>>( 
            partial, in, N );
    Reduction7_kernel<32><<< 
        1, numThreads>>>( 
            answer, partial, numBlocks );
}
