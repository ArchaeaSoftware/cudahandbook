/*
 *
 * testScanWarp.cu
 *
 * Microdemo to test warp scan algorithms.
 *
 * Build with: nvcc -I ..\chLib <options> testScanWarp.cu
 * Requires: No minimum SM requirement.
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

#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <chAssert.h>
#include <chError.h>

typedef unsigned int uint;

#include "scanWarp.cuh"
#include "scanWarp2.cuh"
#include "scanWarpShuffle.cuh"

#define min(a,b) ((a)<(b)?(a):(b))

int *g_hostIn, *g_hostOut;


enum ScanType {
    Inclusive, Exclusive
};

template<int period>
void
ScanExclusiveCPUPeriodic( int *out, const int *in, size_t N )
{
    for ( size_t i = 0; i < N; i += period ) {
        int sum = 0;
        for ( size_t j = 0; j < period; j++ ) {
            int next = in[i+j]; // in case we are doing this in place
            out[i+j] = sum;
            sum += next;
        }
    }
}

template<int period>
void
ScanInclusiveCPUPeriodic( int *out, const int *in, size_t N )
{
    for ( size_t i = 0; i < N; i += period ) {
        int sum = 0;
        for ( size_t j = 0; j < period; j++ ) {
            sum += in[i+j];
            out[i+j] = sum;
        }
    }
}

void
RandomArray( int *out, size_t N, int modulus )
{
    for ( size_t i = 0; i < N; i++ ) {
        out[i] = rand() % modulus;
    }
}

__global__ void
ScanInclusiveGPUWarp( int *out, const int *in, size_t N )
{
    extern __shared__ int sPartials[];
    for ( size_t i = blockIdx.x*blockDim.x;
                 i < N;
                 i += blockDim.x ) {
        sPartials[threadIdx.x] = in[i+threadIdx.x];
        __syncthreads();
        out[i+threadIdx.x] = scanWarp<int,false>( sPartials+threadIdx.x );
    }
}

void
ScanInclusiveGPU( 
    int *out, 
    const int *in, 
    size_t N, 
    int cThreads )
{
    int cBlocks = (int) (N/150);
    if ( cBlocks > 150 ) {
        cBlocks = 150;
    }
    ScanInclusiveGPUWarp<<<cBlocks, cThreads, cThreads*sizeof(int)>>>( 
        out, in, N );
}

__global__ void
ScanInclusiveGPUWarp2( int *out, const int *in, size_t N )
{
    extern __shared__ int sPartials[];
    for ( size_t i = blockIdx.x*blockDim.x;
                 i < N;
                 i += blockDim.x ) {
        sPartials[threadIdx.x] = in[i+threadIdx.x];
        __syncthreads();
        out[i+threadIdx.x] = scanWarp2<int,false>( sPartials+threadIdx.x );
    }
}

void
ScanInclusiveGPU2( 
    int *out, 
    const int *in, 
    size_t N, 
    int cThreads )
{
    int cBlocks = (int) (N/150);
    if ( cBlocks > 150 ) {
        cBlocks = 150;
    }
    ScanInclusiveGPUWarp2<<<cBlocks, cThreads, cThreads*sizeof(int)>>>( 
            out, in, N );
}

__global__ void
ScanInclusiveGPUWarpShuffle( int *out, const int *in, size_t N )
{
    extern __shared__ int sPartials[];
    for ( size_t i = blockIdx.x*blockDim.x;
                 i < N;
                 i += blockDim.x ) {
        out[i+threadIdx.x] = inclusive_scan_warp_shfl<5>( in[i+threadIdx.x] );
    }
}

void
ScanInclusiveGPUShuffle( 
    int *out, 
    const int *in, 
    size_t N, 
    int cThreads )
{
    int cBlocks = (int) (N/150);
    if ( cBlocks > 150 ) {
        cBlocks = 150;
    }
    ScanInclusiveGPUWarpShuffle<<<cBlocks, cThreads>>>( out, in, N );
}

template<class T>
bool
TestScanWarp( 
    float *pMelementspersecond,
    const char *szScanFunction, 
    void (*pfnScanCPU)(T *, const T *, size_t),
    void (*pfnScanGPU)(T *, const T *, size_t, int), 
    size_t N, 
    int numThreads )
{
    bool ret = false;
    cudaError_t status;
    int *inGPU = 0;
    int *outGPU = 0;
    int *inCPU = (T *) malloc( N*sizeof(T) );
    int *outCPU = (int *) malloc( N*sizeof(T) );
    int *hostGPU = (int *) malloc( N*sizeof(T) );
    cudaEvent_t evStart = 0, evStop = 0;
    if ( 0==inCPU || 0==outCPU || 0==hostGPU )
        goto Error;

    printf( "Testing %s (%d threads/block)\n", szScanFunction, numThreads );

    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );
    CUDART_CHECK( cudaMalloc( &inGPU, N*sizeof(T) ) );
    CUDART_CHECK( cudaMalloc( &outGPU, N*sizeof(T) ) );
    CUDART_CHECK( cudaMemset( inGPU, 0, N*sizeof(T) ) );
    CUDART_CHECK( cudaMemset( outGPU, 0, N*sizeof(T) ) );

    CUDART_CHECK( cudaMemset( outGPU, 0, N*sizeof(T) ) );

    RandomArray( inCPU, N, 256 );
for ( int i = 0; i < N; i++ ) {
    inCPU[i] = i;
}
    
    pfnScanCPU( outCPU, inCPU, N );
g_hostIn = inCPU;

    CUDART_CHECK( cudaMemcpy( inGPU, inCPU, N*sizeof(T), cudaMemcpyHostToDevice ) );
    CUDART_CHECK( cudaEventRecord( evStart, 0 ) );
    pfnScanGPU( outGPU, inGPU, N, numThreads );
    CUDART_CHECK( cudaEventRecord( evStop, 0 ) );
    CUDART_CHECK( cudaMemcpy( hostGPU, outGPU, N*sizeof(T), cudaMemcpyDeviceToHost ) );
    for ( size_t i = 0; i < N; i++ ) {
        if ( hostGPU[i] != outCPU[i] ) {
            printf( "Scan failed\n" );
#ifdef _WIN32
            __debugbreak();//_asm int 3
#else
            assert(0);
#endif
            goto Error;
        }
    }
    {
        float ms;
        CUDART_CHECK( cudaEventElapsedTime( &ms, evStart, evStop ) );
        double Melements = N/1e6;
        *pMelementspersecond = 1000.0f*Melements/ms;
    }
    ret = true;
Error:
    cudaEventDestroy( evStart );
    cudaEventDestroy( evStop );
    cudaFree( outGPU );
    cudaFree( inGPU );
    free( inCPU );
    free( outCPU );
    free( hostGPU );
    return ret;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int maxThreads;
    int numInts = 32*1048576;

    CUDART_CHECK( cudaSetDevice( 0 ) );
    CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );

    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, 0 );
        maxThreads = prop.maxThreadsPerBlock;
    }

#define SCAN_TEST_VECTOR( CPUFunction, GPUFunction, N, numThreads ) do { \
    float fMelementsPerSecond; \
    srand(0); \
    bool bSuccess = TestScanWarp<int>( &fMelementsPerSecond, #GPUFunction, CPUFunction, GPUFunction, N, numThreads ); \
    if ( ! bSuccess ) { \
        printf( "%s failed: N=%d, numThreads=%d\n", #GPUFunction, N, numThreads ); \
        exit(1); \
    } \
    if ( fMelementsPerSecond > maxElementsPerSecond ) { \
        maxElementsPerSecond = fMelementsPerSecond; \
    } \
\
} while (0)

    printf( "Problem size: %d integers\n", numInts );

    for ( int numThreads = 256; numThreads <= maxThreads; numThreads *= 2 ) {
        float maxElementsPerSecond = 0.0f;
        SCAN_TEST_VECTOR( ScanInclusiveCPUPeriodic<32>, ScanInclusiveGPU, numInts, numThreads );
        printf( "GPU: %.2f Melements/s\n", maxElementsPerSecond );
        maxElementsPerSecond = 0.0f;
        SCAN_TEST_VECTOR( ScanInclusiveCPUPeriodic<32>, ScanInclusiveGPU2, numInts, numThreads );
        printf( "GPU2: %.2f Melements/s\n", maxElementsPerSecond );
        maxElementsPerSecond = 0.0f;
        SCAN_TEST_VECTOR( ScanInclusiveCPUPeriodic<32>, ScanInclusiveGPUShuffle, numInts, numThreads );
        printf( "Shuffle: %.2f Melements/s\n", maxElementsPerSecond );
    }

    return 0;
Error:
    return 1;
}
