/*
 *
 * mallocSpeed.cu
 *
 * Microbenchmark for overhead and per-page speed of host and
 * device memory allocation.
 *
 * Build with: nvcc -I ../chLib <options> mallocSpeed.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2013, Archaea Software, LLC.
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

#include "chTimer.h"

template <bool bDevice>
float
mallocSpeed( int cIterations, size_t N )
{
    chTimerTimestamp start, stop;
    float ret = 0.0f;
    void **p = (void **) malloc( cIterations*sizeof(void *));
    if ( ! p )
        goto Error;
    memset( p, 0, cIterations*sizeof(void *) );
    
    chTimerGetTime( &start );
    for ( size_t i = 0; i < cIterations; i++ ) {
        if ( bDevice ) {
            if ( cudaSuccess != cudaMalloc( &p[i], N ) ) {
                goto Error;
            }
        }
        else {
            if ( cudaSuccess != cudaMallocHost( &p[i], N ) )
                goto Error;
        }
    }
    chTimerGetTime( &stop );
    ret = chTimerElapsedTime( &start, &stop );
Error:
    if ( p ) {
        for ( size_t i = 0; i < cIterations; i++ ) {
            if ( bDevice ) {
                cudaFree( p[i] );
            }
            else {
                cudaFreeHost( p[i] );
            }
        }
        free( p );
    }
    return ret;
}

template<bool bDevice>
float
mallocSpeed( size_t N )
{
    float ret = 0.0f;
    int cIterations;
    for ( cIterations = 1; ret < 0.5f; cIterations *= 2 ) {
        ret = mallocSpeed<bDevice>( cIterations, N );
        if ( 0.0f == ret ) {
            return mallocSpeed<bDevice>( cIterations/2, N) / (cIterations/2);
        }
    }
    return ret / cIterations;
}


int
main( int argc, char *argv[] )
{
    if ( cudaSuccess != cudaFree(0) ) {
        printf( "Initialization failed\n" );
        exit(1);
    }
    printf( "mallocSpeed (4K device): %.2f us/iteration\n", mallocSpeed<true>(4096)*1e6 );
    printf( "mallocSpeed (1M device): %.2f us/iteration\n", mallocSpeed<true>(1048576)*1e6 );
    printf( "mallocSpeed (4K host): %.2f us/iteration\n", mallocSpeed<false>(4096)*1e6 );
    printf( "mallocSpeed (1M host): %.2f us/iteration\n", mallocSpeed<false>(1048576)*1e6 );
    return 0;

}
