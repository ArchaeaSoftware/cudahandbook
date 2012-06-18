/*
 *
 * spinlockReduction.cu
 *
 * Implements reduction (sum) of double-precision values in terms of a 
 * two-stage algorithm that accumulates partial sums in shared memory,
 * then each block acquires a spinlock on the output value to
 * atomically add its partial sum to the final output.
 *
 * Copyright (C) 2012 by Archaea Software, LLC.  All rights reserved.
 *
 * Build line: nvcc --gpu-architecture sm_20 -I ../chLib spinlockReduction.cu
 * Microbenchmark to measure performance of spin locks.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_20 <options> spinlockReduction.cu
 * Requires: SM 2.0 for 
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

#include <stdio.h>

#include <chAssert.h>
#include <chError.h>
#include <chCommandLine.h>

class cudaSpinlock {
public:
    cudaSpinlock( int *p );
    void acquire();
    void release();
private:
    int *m_p;
};

inline __device__
cudaSpinlock::cudaSpinlock( int *p )
{
    m_p = p;
}

inline __device__ void
cudaSpinlock::acquire( )
{
    while ( atomicCAS( m_p, 0, 1 ) );
}

inline __device__ void
cudaSpinlock::release( )
{
    atomicExch( m_p, 0 );
}

//
// From Reduction SDK sample:
//
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
//
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*) (void *) __smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*) (void *) __smem;
    }
};

template<typename ReductionType, typename T>
__device__ ReductionType
Reduce_block( )
{
    SharedMemory<ReductionType> shared_sum;
    const int tid = threadIdx.x;

    for ( int activeThreads = blockDim.x>>1; 
              activeThreads; 
              activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
            shared_sum[tid] += shared_sum[tid+activeThreads];
        }
        __syncthreads();
    }
    return shared_sum[0];
}

/*
 * Contended spinlock.
 */

__device__ int g_acquireCount;
int *g_pacquireCount;

__global__ void
SumValues_1( double *pSum, int *spinlock, const double *in, size_t N, int *acquireCount )
{
    SharedMemory<double> shared;
    cudaSpinlock globalSpinlock( spinlock );

    for ( size_t i = blockIdx.x*blockDim.x+threadIdx.x; 
                 i < N; 
                 i += blockDim.x*gridDim.x ) {
        shared[threadIdx.x] = in[i];
        __syncthreads();
        double blockSum = Reduce_block<double,double>( );
        __syncthreads();

        if ( threadIdx.x == 0 ) {
            globalSpinlock.acquire( );
            *pSum += blockSum;
            __threadfence();
            globalSpinlock.release( );
        }
    }
}

double
AtomicsPerSecond( size_t N, int cBlocks, int cThreads )
{
    double *d_inputValues;
    double *h_inputValues = 0;
    double *d_sumInputValues;
    double h_sumInputValues;
    double sumInputValues;

    int *d_spinLocks = 0;

    double ret = 0.0;
    double elapsedTime;
    float ms;
    cudaError_t status;
    cudaEvent_t evStart = 0;
    cudaEvent_t evStop = 0;

    CUDART_CHECK( cudaMalloc( &d_spinLocks, 1*sizeof(int) ) );
    CUDART_CHECK( cudaMemset( d_spinLocks, 0, 1*sizeof(int) ) );

    CUDART_CHECK( cudaMalloc( &d_sumInputValues, 1*sizeof(double) ) );
    CUDART_CHECK( cudaMemset( d_sumInputValues, 0, 1*sizeof(double) ) );
    
    CUDART_CHECK( cudaMalloc( &d_inputValues, N*sizeof(double) ) );
    h_inputValues = new double[N];
    if ( ! h_inputValues )
        goto Error;

    sumInputValues = 0.0;
    for ( size_t i = 0; i < N; i++ ) {
        double value = (double) rand();
        h_inputValues[i] = value;
        sumInputValues += value;
    }
    CUDART_CHECK( cudaMemcpy( d_inputValues, 
                              h_inputValues, 
                              N*sizeof(double),
                              cudaMemcpyHostToDevice ) );

    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );

    cudaEventRecord( evStart );
    {
        SumValues_1<<<cBlocks,cThreads,cThreads*sizeof(double)>>>( 
            d_sumInputValues, 
            d_spinLocks, 
            d_inputValues,
            N,
            g_pacquireCount );
        CUDART_CHECK( cudaMemcpy( &h_sumInputValues, 
                                  d_sumInputValues, 
                                  sizeof(double), 
                                  cudaMemcpyDeviceToHost ) );
        if ( h_sumInputValues != sumInputValues ) {
            printf( "Mismatch: %E should be %E\n", h_sumInputValues, sumInputValues );
        }
    }

    cudaEventRecord( evStop );
    CUDART_CHECK( cudaThreadSynchronize() );

    // make configurations that cannot launch error-out with 0 bandwidth
    CUDART_CHECK( cudaGetLastError() ); 

    CUDART_CHECK( cudaEventElapsedTime( &ms, evStart, evStop ) );
    elapsedTime = ms/1000.0f;

	// Return operations per second
    ret = (double) N / elapsedTime;

Error:
    cudaFree( d_spinLocks );
    cudaFree( d_sumInputValues );
    cudaFree( d_inputValues );
    delete[] h_inputValues;

    cudaEventDestroy( evStart );
    cudaEventDestroy( evStop );
    return ret;
}

double
ReportRow( size_t N, size_t threadStart, size_t threadStop, size_t cBlocks, int step )
{
    int maxThreads = 0;
    double maxops = 0.0;
    for ( int cThreads = threadStart; cThreads <= threadStop; cThreads *= 2 ) {
        double ops = AtomicsPerSecond( N, cBlocks, cThreads );
        if ( ops > maxops ) {
            maxops = ops;
            maxThreads = cThreads;
        }
        printf( "%.2f\t", ops/1e6f );
    }
    printf( "%.2f\t%d\n", maxops/1e6f, maxThreads );
    return maxops;
}

void
Shmoo( size_t N, size_t threadStart, size_t threadStop, size_t cBlocks )
{
    printf( "All units in Mops (millions of operations per second):\n" );
    for ( int cThreads = threadStart; cThreads <= threadStop; cThreads *= 2 ) {
        printf( "%d\t", cThreads );
    }
    printf( "maxops\tmaxThreads\n" );
    ReportRow( N, threadStart, threadStop, cBlocks, 1 );
    ReportRow( N, threadStart, threadStop, cBlocks, 2 );
    ReportRow( N, threadStart, threadStop, cBlocks, 4 );
    ReportRow( N, threadStart, threadStop, cBlocks, 8 );
    ReportRow( N, threadStart, threadStop, cBlocks, 16 );
    ReportRow( N, threadStart, threadStop, cBlocks, 32 );
    ReportRow( N, threadStart, threadStop, cBlocks, 64 );
    ReportRow( N, threadStart, threadStop, cBlocks, 128 );
    ReportRow( N, threadStart, threadStop, cBlocks, 256 );
}


int
main( int argc, char *argv[] )
{
    int device = 0;
    int size = 16;
    if ( chCommandLineGet( &device, "device", argc, argv ) ) {
        printf( "Using device %d...\n", device );
    }
    cudaSetDevice(device);
    if ( chCommandLineGet( &size, "size", argc, argv ) ) {
        printf( "Using %dM operands ...\n", size );
    }

    cudaGetSymbolAddress( (void **) &g_pacquireCount, "g_acquireCount" );

    Shmoo(  (size_t) 131072, 32, 1024, 1500 );
    return 0;
}
