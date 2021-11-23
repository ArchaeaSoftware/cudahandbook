/*
 *
 * globalCopy.cu
 *
 * Microbenchmark for copy bandwidth of global memory.
 *
 * Build with: nvcc -I ../chLib <options> globalCopy.cu
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

#include <stdio.h>

#include <chError.h>
#include <chCommandLine.h>

template<class T, const int n> 
__global__ void
GlobalCopy( T *out, const T *in, size_t N )
{
    T temp[n];
    size_t i;
    for ( i = n*blockIdx.x*blockDim.x+threadIdx.x; 
          i < N-n*blockDim.x*gridDim.x; 
          i += n*blockDim.x*gridDim.x ) {
        for ( int j = 0; j < n; j++ ) {
            size_t index = i+j*blockDim.x;
            temp[j] = in[index];
        }
        for ( int j = 0; j < n; j++ ) {
            size_t index = i+j*blockDim.x;
            out[index] = temp[j];
        }
    }
    // to avoid the (index<N) conditional in the inner loop, 
    // we left off some work at the end
    for ( int j = 0; j < n; j++ ) {
        for ( int j = 0; j < n; j++ ) {
            size_t index = i+j*blockDim.x;
            if ( index<N ) temp[j] = in[index];
        }
        for ( int j = 0; j < n; j++ ) {
            size_t index = i+j*blockDim.x;
            if ( index<N ) out[index] = temp[j];
        }
    }
}

template<class T, const int n, bool bOffsetDst, bool bOffsetSrc>
double
BandwidthCopy( T *deviceOut, T *deviceIn,
               T *hostOut, T *hostIn,
               size_t N,
               cudaEvent_t evStart, cudaEvent_t evStop,
               int cBlocks, int cThreads )
{
    double ret = 0.0;
    double elapsedTime;
    float ms;
    int cIterations;
    cudaError_t status;

    for ( int i = 0; i < N; i++ ) {
        int r = rand();
        hostIn[i] = *(T *)(&r); // for small ints, LSBs; for int2 and int4, some stack cruft
    }

    memset( hostOut, 0, N*sizeof(T) );
    cuda(Memcpy( deviceIn, hostIn, N*sizeof(T), cudaMemcpyHostToDevice ) );
    {
        // confirm that kernel launch with this configuration writes correct result
        GlobalCopy<T,n><<<cBlocks,cThreads>>>( 
            deviceOut+bOffsetDst,
            deviceIn+bOffsetSrc,
            N-bOffsetDst-bOffsetSrc );
        cuda(Memcpy( hostOut, deviceOut, N*sizeof(T), cudaMemcpyDeviceToHost ) );
        cuda(GetLastError() ); 
        if ( memcmp( hostOut+bOffsetDst, hostIn+bOffsetSrc, (N-bOffsetDst-bOffsetSrc)*sizeof(T) ) ) {
            printf( "Incorrect copy performed!\n" );
            goto Error;
        }
    }

    cIterations = 10;
    cudaEventRecord( evStart );
    for ( int i = 0; i < cIterations; i++ ) {
        GlobalCopy<T,n><<<cBlocks,cThreads>>>( deviceOut+bOffsetDst, deviceIn+bOffsetSrc, N-bOffsetDst-bOffsetSrc );
    }
    cudaEventRecord( evStop );
    cuda(DeviceSynchronize() );
    // make configurations that cannot launch error-out with 0 bandwidth
    cuda(GetLastError() ); 
    cuda(EventElapsedTime( &ms, evStart, evStop ) );
    elapsedTime = ms/1000.0f;

    // bytes per second
    ret = ((double)2*N*cIterations*sizeof(T)) / elapsedTime;
    // gigabytes per second
    ret /= 1024.0*1048576.0;

Error:
    return ret;
}

template<class T, const int n, bool bOffsetDst, bool bOffsetSrc>
double
ReportRow( size_t N, size_t threadStart, size_t threadStop, size_t cBlocks )
{
    T *deviceIn = 0;
    T *deviceOut = 0;
    T *hostIn = 0;
    T *hostOut = 0;
    cudaEvent_t evStart = 0;
    cudaEvent_t evStop = 0;
    cudaError_t status;

    int maxThreads = 0;
    double maxBW = 0.0;

    cuda(Malloc( &deviceIn, N*sizeof(T) ) );
    cuda(Malloc( &deviceOut, N*sizeof(T) ) );
    cuda(Memset( deviceOut, 0, N*sizeof(T) ) );

    hostIn = new T[N];
    if ( ! hostIn )
        goto Error;
    hostOut = new T[N];
    if ( ! hostOut )
        goto Error;

    cuda(EventCreate( &evStart ) );
    cuda(EventCreate( &evStop ) );

    printf( "%d\t", n );

    for ( int cThreads = threadStart; cThreads <= threadStop; cThreads *= 2 ) {
        double bw = BandwidthCopy<T,n,bOffsetDst,bOffsetSrc>(
            deviceOut, deviceIn, hostOut, hostIn, N,
            evStart, evStop, cBlocks, cThreads );
        if ( bw > maxBW ) {
            maxBW = bw;
            maxThreads = cThreads;
        }
        printf( "%.2f\t", bw );
    }
    printf( "%.2f\t%d\n", maxBW, maxThreads );
Error:
    if ( hostIn ) delete[] hostIn;
    if ( hostOut ) delete[] hostOut;
    cudaEventDestroy( evStart );
    cudaEventDestroy( evStop );
    cudaFree( deviceIn );
    cudaFree( deviceOut );
    return maxBW;
}

template<class T, bool bOffsetDst, bool bOffsetSrc>
void
Shmoo( size_t N, size_t threadStart, size_t threadStop, size_t cBlocks )
{
    printf( "Operand size: %d byte%c\n", (int) sizeof(T), sizeof(T)==1 ? '\0' : 's' );
    printf( "Input size: %dM operands\n", (int) (N>>20) );
    printf( "                      Block Size\n" );
    printf( "Unroll\t" );
    for ( int cThreads = threadStart; cThreads <= threadStop; cThreads *= 2 ) {
        printf( "%d\t", cThreads );
    }
    printf( "maxBW\tmaxThreads\n" );
    ReportRow<T, 1, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 2, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 3, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 4, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 5, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 6, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 7, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 8, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T, 9, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T,10, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T,11, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T,12, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T,13, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T,14, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T,15, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
    ReportRow<T,16, bOffsetDst, bOffsetSrc >( N, threadStart, threadStop, cBlocks );
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

    if ( chCommandLineGetBool( "uncoalesced_read", argc, argv ) ) {
        if ( chCommandLineGetBool( "uncoalesced_write", argc, argv ) ) {
            printf( "Using uncoalesced reads and writes\n" );
            Shmoo< char, true, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<short, true, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<  int, true, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int2, true, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int4, true, true>( (size_t) size*1048576, 32, 512, 150 );
        }
        else {
            printf( "Using coalesced writes and uncoalesced reads\n" );
            Shmoo< char,false, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<short,false, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<  int,false, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int2,false, true>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int4,false, true>( (size_t) size*1048576, 32, 512, 150 );
        }
    } else {
        if ( chCommandLineGetBool( "uncoalesced_write", argc, argv ) ) {
            printf( "Using uncoalesced writes and coalesced reads\n" );
            Shmoo< char, true,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<short, true,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<  int, true,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int2, true,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int4, true,false>( (size_t) size*1048576, 32, 512, 150 );
        }
        else {
            printf( "Using coalesced reads and writes\n" );
            Shmoo< char,false,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<short,false,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo<  int,false,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int2,false,false>( (size_t) size*1048576, 32, 512, 150 );
            Shmoo< int4,false,false>( (size_t) size*1048576, 32, 512, 150 );
        }
    }
    return 0;
}
