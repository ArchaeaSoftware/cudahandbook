/*
 *
 * AddKernel.cuh
 *
 * CUDA header to implement a 'makework' kernel that iterates a specified
 * number of times.  Other versions of this kernel 'check in' and 
 * 'check out,' using atomic OR to show which kernels are concurrently
 * active and atomicMax to track the maximum number of concurrently-
 * active kernels.
 *
 * Included by:
 *     concurrencyKernelKernel.cu
 *     concurrencyKernelMapped.cu
 *     concurrencyMemcpyKernel.cu
 *     concurrencyMemcpyKernelMapped.cu
 *     TimeConcurrentKernelKernel.cuh
 *     TimeConcurrentKernelMapped.cuh
 *     TimeConcurrentMemcpyKernel.cuh
 *     TimeSequentialKernelKernel.cuh
 *     TimeSequentialMemcpyKernelMapped.cuh
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

#ifndef __CUDAHANDBOOK__ADD_KERNEL__
#define __CUDAHANDBOOK__ADD_KERNEL__

//
// simple AddKernel with no loop unroll and no concurrency tracking
//
__global__ void
AddKernel( int *out, const int *in, size_t N, int increment, int cycles )
{
    for ( size_t i = blockIdx.x*blockDim.x+threadIdx.x; 
                 i < N;
                 i += blockDim.x*gridDim.x )
    {
        volatile int value = in[i];
        for ( int j = 0; j < cycles; j++ ) {
            value += increment;
        }
        out[i] = value;
    }
}

//
// Structure used by the next AddKernel to report observed concurrency.
// Change g_maxStreams to the desired maximum number of supported streams.
//

static const int g_maxStreams = 8;
typedef struct KernelConcurrencyData_st {
    int mask;       // mask of active kernels
    int masks[g_maxStreams];

    int count;      // number of active kernels
    int countMax;   // atomic max of kernel count

    int counts[g_maxStreams];
} KernelConcurrencyData;

void
PrintKernelData( const KernelConcurrencyData& kernelData )
{
    printf( "Kernel data:\n" );

    printf( "    Masks: ( " );
    for ( int i = 0; i < 8; i++ ) {
        printf( " 0x%x ", kernelData.masks[i] );
    }
    printf( ")\n" );

    printf( "    Up to %d kernels were active: (", kernelData.countMax );
    for ( int i = 0; i < 8; i++ ) {
        printf( "0x%x ", kernelData.counts[i] );
    }
    printf( ")\n" );
}

template<const int unrollFactor>
__device__ void
AddKernel( int *out, const int *in, size_t N, int increment, int cycles )
{
    for ( size_t i = unrollFactor*blockIdx.x*blockDim.x+threadIdx.x; 
                 i < N;
                 i += unrollFactor*blockDim.x*gridDim.x )
    {
        int values[unrollFactor];

        for ( int iUnroll = 0; iUnroll < unrollFactor; iUnroll++ ) {
            size_t index = i+iUnroll*blockDim.x;
            values[iUnroll] = in[index];
        }
        for ( int iUnroll = 0; iUnroll < unrollFactor; iUnroll++ ) {
            for ( int k = 0; k < cycles; k++ ) {
                values[iUnroll] += increment;
            }
        }
        for ( int iUnroll = 0; iUnroll < unrollFactor; iUnroll++ ) {
            size_t index = i+iUnroll*blockDim.x;
            out[index] = values[iUnroll];
        }
    }
}

//
// Functionally identical to earlier AddKernel, but with inner loops
// unrolled by the unrollFactor template parameter
//
// This kernel is called by the __global__ function AddKernel that
// switches on the unroll factor.
//
// The kid ("kernel ID") parameter specifies the index into the
// masks and counts arrays in KernelConcurrencyData.  Each kernel
// records its view of the shared global, to cross-check against
// the eventual reported maximum.
// 
//

__device__ KernelConcurrencyData g_kernelData;

template<const int unrollFactor>
__device__ void
AddKernel_helper( int *out, const int *in, size_t N, int increment, int cycles, 
    int kid, KernelConcurrencyData *kernelData )
{
    // check in, and record active kernel mask and count as seen by this kernel.
    if ( kernelData && blockIdx.x==0 && threadIdx.x == 0 ) {
        int myMask = atomicOr( &kernelData->mask, 1<<kid );
        kernelData->masks[kid] = myMask | (1<<kid);

        int myCount = atomicAdd( &kernelData->count, 1 );
        atomicMax( &kernelData->countMax, myCount+1 );
        kernelData->counts[kid] = myCount+1;
    }

    for ( size_t i = unrollFactor*blockIdx.x*blockDim.x+threadIdx.x; 
                 i < N;
                 i += unrollFactor*blockDim.x*gridDim.x )
    {
        int values[unrollFactor];

        for ( int iUnroll = 0; iUnroll < unrollFactor; iUnroll++ ) {
            size_t index = i+iUnroll*blockDim.x;
            values[iUnroll] = in[index];
        }
        for ( int iUnroll = 0; iUnroll < unrollFactor; iUnroll++ ) {
            for ( int k = 0; k < cycles; k++ ) {
                values[iUnroll] += increment;
            }
        }
        for ( int iUnroll = 0; iUnroll < unrollFactor; iUnroll++ ) {
            size_t index = i+iUnroll*blockDim.x;
            out[index] = values[iUnroll];
        }
    }
    // check out
    if ( kernelData && blockIdx.x==0 && threadIdx.x==0 ) {
        atomicAnd( &kernelData->mask, ~(1<<kid) );
        atomicAdd( &kernelData->count, -1 );
    }
}

//
// The non-templatized version of AddKernel
//
__global__ void
AddKernel( int *out, const int *in, size_t N, int increment, 
    int cycles, int kid, KernelConcurrencyData *kernelData, int unrollFactor )
{
    switch ( unrollFactor ) {
        case 1: return AddKernel_helper<1>( out, in, N, increment, cycles, kid, kernelData );
        case 2: return AddKernel_helper<2>( out, in, N, increment, cycles, kid, kernelData );
        case 4: return AddKernel_helper<4>( out, in, N, increment, cycles, kid, kernelData );
    }
}

#endif
