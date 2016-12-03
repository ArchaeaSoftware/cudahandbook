/*
 *
 * testFunnelShift.cu
 *
 * Microdemo to illustrate the workings of Kepler's new funnel shift instruction.
 * 
 * Build with: nvcc -I ..\chLib <options> testFunnelShift.cu
 * Requires: SM 3.5 or higher.
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

#include <stdint.h>
#include <stdio.h>

#include <sm_35_intrinsics.h>


#define SHIFT_FLAGS_LEFT	0x00
#define SHIFT_FLAGS_RIGHT	0x01
#define SHIFT_FLAGS_CLAMP	0x02
#define SHIFT_FLAGS_MASK	0x03

template<bool bClamp>
uint32_t
FunnelShiftLeft( uint32_t a, uint32_t b, uint32_t shift )
{
    shift = ( bClamp ) ? min(shift, 32) : shift & 31;
    return (b << shift) | (a >> (32-shift));
}

template<bool bClamp>
uint32_t
FunnelShiftRight( uint32_t a, uint32_t b, uint32_t shift )
{
    shift = ( bClamp ) ? min(shift, 32) : shift & 31;
    return (b << (32-shift)) | (a >> shift);
}

template<unsigned int Flags>
uint32_t
FunnelShift( uint32_t a, uint32_t b, uint32_t shift )
{
    if ( Flags & SHIFT_FLAGS_RIGHT ) {
        return FunnelShiftRight<0!=(Flags&SHIFT_FLAGS_CLAMP)>( a, b, shift );
    }
    else {
        return FunnelShiftLeft<0!=(Flags&SHIFT_FLAGS_CLAMP)>( a, b, shift );
    }
}


template<unsigned int Flags>
__global__ void
TestFunnelShift( int *p )
{
    if ( blockIdx.x == 0 && threadIdx.x == 0 ) {
        if ( Flags & SHIFT_FLAGS_RIGHT ) {
            if ( Flags & SHIFT_FLAGS_CLAMP ) {
                p[0] = __funnelshift_rc( p[1], p[2], p[3] );
            }
            else {
                p[0] = __funnelshift_r( p[1], p[2], p[3] );
            }
        }
        else {
            if ( Flags & SHIFT_FLAGS_CLAMP ) {
                p[0] = __funnelshift_lc( p[1], p[2], p[3] );
            }
            else {
                p[0] = __funnelshift_l( p[1], p[2], p[3] );
            }
        }
    }
}

cudaError_t
DoFunnelShift( int *out, int lo, int hi, int shift, bool bRight, bool bClamp )
{
    int *hptr = 0;
    int *dptr = 0;
    int emulatedValue;
    cudaError_t status;

    cuda(HostAlloc( &hptr, 4*sizeof(int), cudaHostAllocMapped ) );
    cuda(HostGetDevicePointer( &dptr, hptr, 0 ) );
    hptr[1] = lo;
    hptr[2] = hi;
    hptr[3] = shift;
    if ( bRight ) {
        if ( bClamp ) {
            emulatedValue = FunnelShift<SHIFT_FLAGS_RIGHT|SHIFT_FLAGS_CLAMP>( lo, hi, shift );
            TestFunnelShift<SHIFT_FLAGS_RIGHT|SHIFT_FLAGS_CLAMP><<<1,1>>>( dptr );
        }
        else {
            emulatedValue = FunnelShift<SHIFT_FLAGS_RIGHT>( lo, hi, shift );
            TestFunnelShift<SHIFT_FLAGS_RIGHT><<<1,1>>>( dptr );
        }
    }
    else {
        if ( bClamp ) {
            emulatedValue = FunnelShift<SHIFT_FLAGS_LEFT|SHIFT_FLAGS_CLAMP>( lo, hi, shift );
            TestFunnelShift<SHIFT_FLAGS_LEFT|SHIFT_FLAGS_CLAMP><<<1,1>>>( dptr );
        }
        else {
            emulatedValue = FunnelShift<SHIFT_FLAGS_LEFT>( lo, hi, shift );
            TestFunnelShift<SHIFT_FLAGS_LEFT><<<1,1>>>( dptr );
        }
    }
    cuda(DeviceSynchronize() );
    printf( "d value: 0x%x (%d decimal)\n", emulatedValue, emulatedValue );
    *out = hptr[0];
Error:
    cudaFreeHost( hptr );
    return status;
}

int
getint( int base )
{
    char s[256];
    char *p = fgets( s, 255, stdin );
    if ( p != s )
        return 0;
    return (int) strtol( s, &p, base );
}

int
main( int argc, char *argv[] )
{
    int ret = 1;
    cudaError_t status;
    int out, lo, hi, shift, right, clamp;

    while (1) {
        printf( "lo (hex): " ); lo = getint(16); printf( "Got 0x%x\n", lo );
        printf( "hi (hex): " ); hi = getint(16);
        printf( "shift (decimal - enter 0 to exit): " ); shift = getint(10);
        if ( 0==shift ) {
            printf( "Zero input - exiting\n" );
            return 0;
        }
        printf( "Shift right (nonzero means yes)? " ); right = getint( 10 );
        printf( "Clamp (nonzero means yes)? " ); clamp = getint( 10 );
        CUDART_CHECK( DoFunnelShift( &out, lo, hi, shift, 0!=right, 0!=clamp ) );
        printf( "Result: 0x%x (%d decimal)\n", out, out );
    }
Error:
    printf( "Error %d (%s)\n", status, cudaGetErrorString( status ) );
    return ret;
}
