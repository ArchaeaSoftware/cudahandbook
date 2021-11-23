/*
 *
 * testShuffle.cu
 *
 * Microdemo to illustrate the workings of Kepler's new shuffle instruction.
 * 
 * Build with: nvcc -I ..\chLib <options> testShuffle.cu
 * Requires: SM 3.0 or higher.
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

#include <chError.h>

#include <stdio.h>

#include <sm_30_intrinsics.h>

__global__ void
TestShuffle( int *out, const int *in, size_t N  )
{
    size_t i = blockIdx.x*blockDim.x+threadIdx.x;

    int value = (int) i;//in[i];
    out[i] = __shfl_up_sync( 0xffffffff, value, 1 );
}

cudaError_t
PrintShuffle( int offset, size_t cInts )
{
    int *dptr = 0;
    cudaError_t status;
    int h[64];
    cuda(Malloc( &dptr, cInts*sizeof(int) ) );
    TestShuffle<<<1,cInts>>>( dptr, dptr, cInts );
    cuda(Memcpy( h, dptr, cInts*sizeof(int), cudaMemcpyDeviceToHost ) );
    for ( size_t i = 0; i < cInts; i++ ) {
        printf( "%3x", h[i] );
        if (31==i%32) printf("\n");
    }
    printf( "\n" );
Error:
    cudaFree( dptr );
    return status;
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    int cInts = 64;
    cudaError_t status;

    CUDART_CHECK( PrintShuffle( 1, cInts ) );
    return 0;
Error:
    printf( "Error %d (%s)\n", status, cudaGetErrorString( status ) );
    return ret;
}
