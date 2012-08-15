/*
 *
 * AOStoSOA_1.cuh
 *
 * Header for microdemo that illustrates how to convert from AOS (array
 * of structures) to SOA (structure of arrays) representation.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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

template<typename T, const int k>
__global__ void
AOStoSOA_2( T **_out, const T *in, size_t N )
{
    extern __shared__ T s[];
    T *out[k];
    for ( int i = 0; i < k; i++ ) {
        out[i] = _out[i];
    }

    for ( size_t i = blockIdx.x*blockDim.x;
                 i < N;
                 i += blockDim.x*gridDim.x ) {

        for ( size_t j = threadIdx.x; j < k*blockDim.x; j += blockDim.x ) {
            size_t indexIn = i*k+j;
            if ( indexIn < k*N ) {
                s[j] = in[indexIn];
            }
        }
        __syncthreads();

        for ( int j = 0; j < k; j++ ) {
            size_t indexOut = i+threadIdx.x;
            if ( indexOut < N ) {
                out[j][indexOut] = s[k*threadIdx.x+j];
            }
        }
        __syncthreads();
    }
}

template<typename T, const int k>
void
AOStoSOA_2( T **out, const T *in, size_t N, int cBlocks, int cThreads )
{
    AOStoSOA_2<T,k><<<cBlocks, cThreads,k*cThreads*sizeof(T)>>>( out, in, N );
}
