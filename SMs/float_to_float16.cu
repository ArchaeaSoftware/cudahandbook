/*
 *
 * float_to_float16.cu
 *
 * Test every float32 value and check that our conversion routine
 * agrees with the float->float16 intrinsic in CUDA.
 *
 * Build with: nvcc -I ..\chLib <options> float_to_float16.cu
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

const size_t sizeComparisonArray = 128*1048576;

__global__ void
ConvertFloatRange( unsigned short *out, int start, int N )
{
    for ( int i =  threadIdx.x+blockIdx.x*blockDim.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        float f = __int_as_float( start+i );
        unsigned short f16 = __float2half_rn( f );
        out[ i ] = f16;
    }
}

// Make mask out of bit count
#define LG_MAKE_MASK(bits) ((1<<bits)-1)

/*
 * exponent shift and mantissa bit count are the same.
 *    When we are shifting, we use [f16|f32]ExpShift
 *    When referencing the number of bits in the mantissa, 
 *        we use [f16|f32]MantissaBits
 */
const int f16ExpShift = 10;
const int f16MantissaBits = 10;

const int f16ExpBias = 15;
const int f16MinExp = -14;
const int f16MaxExp = 15;
const int f16SignMask = 0x8000;

const int f32ExpShift = 23;
const int f32MantissaBits = 23;
const int f32ExpBias = 127;
const int f32SignMask = 0x80000000;

unsigned short 
ConvertFloatToHalf( float f ) 
{
    /*
     * Use a volatile union to portably coerce 
     * 32-bit float into 32-bit integer
     */
    volatile union {
        float f;
        unsigned int u;
    } uf;
    uf.f = f;

    // return value: start by propagating the sign bit.
    unsigned short w = (uf.u >> 16) & f16SignMask;
    
    // Extract input magnitude and exponent
    unsigned int mag = uf.u & ~f32SignMask;
    int exp = (int) (mag >> f32ExpShift) - f32ExpBias;

    // Handle float32 Inf or NaN
    if ( exp == f32ExpBias+1 ) {    // INF or NaN

        if ( mag & LG_MAKE_MASK(f32MantissaBits) )
            return 0x7fff; // NaN

        // INF - propagate sign
        return w|0x7c00;
    }

    /*
     * clamp float32 values that are not representable by float16
     */
    {
        // min float32 magnitude that rounds to float16 infinity

        unsigned int f32MinRInfin = (f16MaxExp+f32ExpBias) << 
            f32ExpShift;
        f32MinRInfin |= LG_MAKE_MASK( f16MantissaBits+1 ) << 
            (f32MantissaBits-f16MantissaBits-1);

        if (mag > f32MinRInfin)
            mag = f32MinRInfin;
    }

    {
        // max float32 magnitude that rounds to float16 0.0

        unsigned int f32MaxRf16_zero = f16MinExp+f32ExpBias-
            (f32MantissaBits-f16MantissaBits-1);
        f32MaxRf16_zero <<= f32ExpShift;
        f32MaxRf16_zero |= LG_MAKE_MASK( f32MantissaBits );

        if (mag < f32MaxRf16_zero) 
            mag = f32MaxRf16_zero;
    }
    
    /*
     * compute exp again, in case mag was clamped above
     */
    exp = (mag >> f32ExpShift) - f32ExpBias;

    // min float32 magnitude that converts to float16 normal
    unsigned int f32Minf16Normal = ((f16MinExp+f32ExpBias)<<
        f32ExpShift);
    f32Minf16Normal |= LG_MAKE_MASK( f32MantissaBits );
    if ( mag >= f32Minf16Normal ) { 
        //
        // Case 1: float16 normal
        //

        // Modify exponent to be biased for float16, not float32
        mag += (unsigned int) ((f16ExpBias-f32ExpBias)<<
            f32ExpShift);

        int RelativeShift = f32ExpShift-f16ExpShift;

        // add rounding bias
        mag += LG_MAKE_MASK(RelativeShift-1);

        // round-to-nearest even
        mag += (mag >> RelativeShift) & 1;

        w |= mag >> RelativeShift; 
    } 
    else { 
        /*
         * Case 2: float16 denormal
         */

        // mask off exponent bits - now fraction only
        mag &= LG_MAKE_MASK(f32MantissaBits);

        // make implicit 1 explicit
        mag |= (1<<f32ExpShift);

        int RelativeShift = f32ExpShift-f16ExpShift+f16MinExp-exp; 

        // add rounding bias
        mag += LG_MAKE_MASK(RelativeShift-1);
        
        // round-to-nearest even
        mag += (mag >> RelativeShift) & 1;

        w |= mag >> RelativeShift;
    } 
    return w; 
}

int
main()
{
    cudaError_t status;
    unsigned short *convertedFloats = 0;
    unsigned short *deviceFloats = 0;
    int cDifferent = 0;

    int numRounds = (int) (4294967296LL / sizeComparisonArray);

    convertedFloats = new unsigned short[sizeComparisonArray];
    if ( ! convertedFloats ) {
        return 1;
    }
    cuda(Malloc( &deviceFloats, 
        sizeComparisonArray*sizeof(*deviceFloats ) ) );
    for ( int i = 0; i < numRounds; i++ ) {
        ConvertFloatRange<<<32,384>>>( 
            deviceFloats, 
            (int) ((size_t)i*sizeComparisonArray), 
            sizeComparisonArray );
        cuda(Memcpy( 
            convertedFloats, 
            deviceFloats, 
            sizeComparisonArray*sizeof(unsigned short), 
            cudaMemcpyDeviceToHost ) );
        cDifferent = 0;
        for ( int j = 0; j < sizeComparisonArray; j++ ) {
            volatile union {
                unsigned int u;
                float f;
            } uf;
            
            uf.u = (unsigned int) ((size_t)i*sizeComparisonArray+j);
            unsigned short f16;
            f16 = ConvertFloatToHalf( uf.f );
            if ( f16 != convertedFloats[j] ) {
                cDifferent += 1;
                if ( cDifferent < 16 ) {
                    printf( "Float 0x%08x converted to %08x, expected %08x\n", uf.u, f16, convertedFloats[j] ); fflush(stdout);
                }
                ConvertFloatToHalf( uf.f );
            }
        }
        printf( "i = %d: cDifferent = %d\n", i, cDifferent );
    }
    printf( "Done: success\n" );
    return 0;
Error:
    printf( "Error\n" );
    return 1;
}


