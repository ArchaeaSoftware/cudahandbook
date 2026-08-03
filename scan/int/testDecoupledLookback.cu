/*
 * testDecoupledLookback.cu
 *
 * Correctness + throughput check for scanDecoupledLookback.cuh. Needs chLib on
 * the include path (for chError.h), as the other scan samples do:
 *
 *     nvcc -O3 -I../../chLib testDecoupledLookback.cu -o testDecoupledLookback
 *     ./testDecoupledLookback [N] [blockSize]
 *
 * Copyright (c) 2016-2026, Archaea Software, LLC. All rights reserved.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <chError.h>

#include "scanDecoupledLookback.cuh"

#define CHECK(e) do { cudaError_t s_ = (e); if ( s_ != cudaSuccess ) { \
    printf( "CUDA error %s at %s:%d\n", cudaGetErrorString(s_), __FILE__, __LINE__ ); \
    return 1; } } while (0)

int
main( int argc, char *argv[] )
{
    size_t N = ( argc > 1 ) ? (size_t) atoll( argv[1] ) : ( 1u << 24 );  // 16M
    int    b = ( argc > 2 ) ? atoi( argv[2] ) : 256;

    int gpu = 0;
    cudaDeviceProp prop;
    CHECK( cudaGetDevice( &gpu ) );
    CHECK( cudaGetDeviceProperties( &prop, gpu ) );
    printf( "GPU: %s (sm_%d%d)   N=%zu   blockSize=%d\n",
            prop.name, prop.major, prop.minor, N, b );

    int *h_in  = (int *) malloc( N * sizeof(int) );
    int *h_out = (int *) malloc( N * sizeof(int) );
    int *h_ref = (int *) malloc( N * sizeof(int) );
    if ( ! h_in || ! h_out || ! h_ref ) { printf( "host OOM\n" ); return 1; }

    srand( 1 );
    for ( size_t i = 0; i < N; i++ )
        h_in[i] = ( rand() & 0xff ) - 0x7f;          // small signed ints

    // serial reference: inclusive scan
    {
        int sum = 0;
        for ( size_t i = 0; i < N; i++ ) { sum += h_in[i]; h_ref[i] = sum; }
    }

    int *d_in = 0, *d_out = 0;
    CHECK( cudaMalloc( &d_in,  N * sizeof(int) ) );
    CHECK( cudaMalloc( &d_out, N * sizeof(int) ) );
    CHECK( cudaMemcpy( d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice ) );

    // correctness
    scanDecoupledLookback<int>( d_out, d_in, N, b );
    CHECK( cudaDeviceSynchronize() );
    CHECK( cudaMemcpy( h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost ) );

    size_t mism = 0;
    for ( size_t i = 0; i < N; i++ ) {
        if ( h_out[i] != h_ref[i] ) {
            if ( mism < 5 )
                printf( "  mismatch @ %zu: got %d, expected %d\n",
                        i, h_out[i], h_ref[i] );
            mism++;
        }
    }
    printf( "Correctness: %s  (%zu mismatches of %zu)\n",
            mism ? "FAIL" : "PASS", mism, N );

    // throughput (rough: the status buffer is re-allocated each call, so this
    // slightly understates a production scan that reuses it)
    cudaEvent_t start, stop;
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );
    const int iters = 50;
    scanDecoupledLookback<int>( d_out, d_in, N, b );   // warm up
    CHECK( cudaDeviceSynchronize() );
    CHECK( cudaEventRecord( start ) );
    for ( int i = 0; i < iters; i++ )
        scanDecoupledLookback<int>( d_out, d_in, N, b );
    CHECK( cudaEventRecord( stop ) );
    CHECK( cudaEventSynchronize( stop ) );
    float ms = 0.0f;
    CHECK( cudaEventElapsedTime( &ms, start, stop ) );
    ms /= iters;
    double gbps  = ( 2.0 * N * sizeof(int) ) / ( ms * 1e-3 ) / 1e9;  // read + write
    double gelem = N / ( ms * 1e-3 ) / 1e9;
    printf( "Throughput: %.3f ms/scan   %.1f GB/s   %.2f Gelem/s\n",
            ms, gbps, gelem );

    free( h_in ); free( h_out ); free( h_ref );
    cudaFree( d_in ); cudaFree( d_out );
    return mism ? 1 : 0;
}
