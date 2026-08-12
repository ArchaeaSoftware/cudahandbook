/*
 *
 * testScanWarp.cu
 *
 * Exercises the warp/block scan policies in scanWarpPolicy.cuh. Every policy is
 * instantiated (and therefore compiled) here and selected purely by template
 * argument, so no variant can silently bit-rot. Each is cross-checked exact
 * against a host inclusive scan and timed.
 *
 * Build (via CMake) or: nvcc -I ../../chLib -arch=sm_86 testScanWarp.cu
 *
 * Copyright (c) 2025-2026, Archaea Software, LLC.
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

#include <stdio.h>
#include <stdlib.h>

#include <chError.h>

#include "../scanWarpPolicy.cuh"

static const int BLOCKSIZE = 256;   // 8 warps

template<class Policy>
__global__ void
warpScanKernel( const int *in, int *out, int N )
{
    extern __shared__ int s[];
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += gridDim.x*blockDim.x )
        out[i] = warpScanInclusive<Policy,int>( in[i], s + threadIdx.x );
}

template<class Policy>
__global__ void
blockScanKernel( const int *in, int *out, int N )
{
    extern __shared__ int s[];
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += gridDim.x*blockDim.x )
        out[i] = blockScanInclusive<Policy,int>( in[i], s );
}

// Launch, cross-check exact against a host inclusive scan over each group
// (group = 32 for warp, BLOCKSIZE for block), and time. Returns false on mismatch.
template<class Policy, bool BLOCK>
static bool
runOne( const int *dIn, int *dOut, const int *hIn, int *hOut, int N )
{
    dim3 grid( (N + BLOCKSIZE - 1) / BLOCKSIZE );
    auto launch = [&] {
        if ( BLOCK ) blockScanKernel<Policy><<<grid,BLOCKSIZE,BLOCKSIZE*sizeof(int)>>>( dIn, dOut, N );
        else         warpScanKernel <Policy><<<grid,BLOCKSIZE,BLOCKSIZE*sizeof(int)>>>( dIn, dOut, N );
    };
    launch();
    if ( cudaSuccess != cudaDeviceSynchronize() ) { printf( "  %-16s  launch failed\n", Policy::name ); return false; }
    cudaMemcpy( hOut, dOut, N*sizeof(int), cudaMemcpyDeviceToHost );

    const int group = BLOCK ? BLOCKSIZE : 32;
    long bad = 0;
    for ( int b = 0; b < N; b += group ) {
        int acc = 0;
        for ( int j = 0; j < group && b+j < N; j++ ) {
            acc += hIn[b+j];
            if ( hOut[b+j] != acc ) bad++;
        }
    }

    cudaEvent_t e0, e1;
    cudaEventCreate( &e0 );
    cudaEventCreate( &e1 );
    const int iters = 200;
    cudaEventRecord( e0, 0 );
    for ( int i = 0; i < iters; i++ ) launch();
    cudaEventRecord( e1, 0 );
    cudaEventSynchronize( e1 );
    float ms = 0.f;
    cudaEventElapsedTime( &ms, e0, e1 );
    ms /= iters;
    cudaEventDestroy( e0 );
    cudaEventDestroy( e1 );

    printf( "  %-16s  %-5s  %-4s  %6.1f GB/s\n",
            Policy::name, BLOCK ? "block" : "warp", bad ? "FAIL" : "PASS",
            2.0*N*sizeof(int)/(ms*1e6) );
    return 0 == bad;
}

int
main()
{
    cudaError_t status_cudart;
    int ret = 1;
    const int N = BLOCKSIZE * 65536;            // 16.7M ints
    int *hIn = NULL, *hOut = NULL, *dIn = NULL, *dOut = NULL;
    bool ok = true;

    hIn  = (int *) malloc( N*sizeof(int) );
    hOut = (int *) malloc( N*sizeof(int) );
    if ( ! hIn || ! hOut ) goto Error_cudart;
    srand( 5 );
    for ( int i = 0; i < N; i++ ) hIn[i] = rand() & 15;

    cuda( Malloc( &dIn,  N*sizeof(int) ) );
    cuda( Malloc( &dOut, N*sizeof(int) ) );
    cuda( Memcpy( dIn, hIn, N*sizeof(int), cudaMemcpyHostToDevice ) );

    {
        cudaDeviceProp prop;
        cuda( GetDeviceProperties( &prop, 0 ) );
        printf( "%s  warp/block scan policies (all live, selected at compile time), N=%d\n\n", prop.name, N );
        printf( "  %-16s  %-5s  %-4s  %s\n", "policy", "level", "chk", "throughput" );
    }

    ok &= runOne<WarpScanShared, false>( dIn, dOut, hIn, hOut, N );
    ok &= runOne<WarpScanShuffle,false>( dIn, dOut, hIn, hOut, N );
    printf( "\n" );
    ok &= runOne<WarpScanShared, true >( dIn, dOut, hIn, hOut, N );
    ok &= runOne<WarpScanShuffle,true >( dIn, dOut, hIn, hOut, N );

    printf( "\n%s\n", ok ? "All policies PASS." : "FAILURES above." );
    ret = ok ? 0 : 1;

Error_cudart:
    cudaFree( dIn );
    cudaFree( dOut );
    free( hIn );
    free( hOut );
    return ret;
}
