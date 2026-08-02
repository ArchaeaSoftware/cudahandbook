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
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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
#include <cuda_fp16.h>

#include <chError.h>

const size_t sizeComparisonArray = 128*1048576;

__global__ void
ConvertFloatRange( unsigned short *out, int start, int N )
{
    for ( int i =  threadIdx.x+blockIdx.x*blockDim.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        float f = __int_as_float( start+i );
        unsigned short f16 = __half_as_ushort( __float2half_rn( f ) );
        out[ i ] = f16;
    }
}

// Make mask out of bit count
#define LG_MAKE_MASK(bits) ((1<<(bits))-1)

/*
 * exponent shift and mantissa bit count are the same.
 *    When we are shifting, we use [f16|f32]ExpShift
 *    When referencing the number of bits in the mantissa, 
 *        we use [f16|f32]MantissaBits
 */
const int f16ExpShift = 10;
const int f16MantissaBits = 10;
const int f16ExpBits = 5;

const int f16ExpBias = 15;
const int f16MinExp = -14;
const int f16MaxExp = 15;
const int f16SignMask = 0x8000;

const int f32ExpShift = 23;
const int f32MantissaBits = 23;
const int f32ExpBits = 8;
const int f32ExpBias = 127;
const int f32SignMask = 0x80000000;

// All bits of the exponent field set.  For float16, this is
// also the encoding of infinity.
const int f16ExpMask = LG_MAKE_MASK(f16ExpBits) << f16ExpShift;
const int f32ExpMask = LG_MAKE_MASK(f32ExpBits) << f32ExpShift;

// Smallest float32 exponent field whose values may round to a
// nonzero float16 (halfway between 0 and the smallest denormal)
const int f32MinRNonzero = (f16MinExp-f16MantissaBits-1+f32ExpBias) <<
    f32ExpShift;

/*
 * Adapted from code posted to the NVIDIA Developer Forums by
 * Norbert Juffa, with hard-coded constants replaced by the
 * symbolic constants defined above:
 *
 * https://forums.developer.nvidia.com/t/error-when-trying-to-use-half-fp16/39786/10
 *
 * Copyright (c) 2015, Norbert Juffa
 * All rights reserved.
 * (distributed under the same 2-clause BSD license as this file)
 */
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

    unsigned int ia = uf.u;
    unsigned short ir;

    // start by propagating the sign bit
    ir = (ia >> 16) & f16SignMask;

    if ( (ia & f32ExpMask) == f32ExpMask ) {    // INF or NaN
        if ( (ia & ~f32SignMask) == f32ExpMask ) {
            // INF - propagate sign
            ir |= f16ExpMask;
        }
        else {
            // canonical NaN
            ir = f16ExpMask | LG_MAKE_MASK(f16MantissaBits);
        }
    }
    else if ( (ia & f32ExpMask) >= (unsigned int) f32MinRNonzero ) {
        int shift = (int) ((ia >> f32ExpShift) &
            LG_MAKE_MASK(f32ExpBits)) - f32ExpBias;
        if ( shift > f16MaxExp ) {
            // overflow - round to infinity
            ir |= f16ExpMask;
        }
        else {
            // extract mantissa and make implicit 1 explicit
            ia = (ia & LG_MAKE_MASK(f32MantissaBits)) |
                (1<<f32ExpShift);
            if ( shift < f16MinExp ) {
                /*
                 * Case 1: float16 denormal
                 */
                int RelativeShift = f32ExpShift-f16ExpShift+
                    f16MinExp-shift;
                ir |= ia >> RelativeShift;
                ia = ia << (32 - RelativeShift);
            }
            else {
                /*
                 * Case 2: float16 normal
                 */
                int RelativeShift = f32ExpShift-f16ExpShift;
                ir |= ia >> RelativeShift;
                ia = ia << (32 - RelativeShift);
                // f16ExpBias-1, since the explicit 1 shifted
                // into the exponent field adds 1 to the exponent
                ir = ir + ((f16ExpBias-1+shift) << f16ExpShift);
            }
            /*
             * Round to nearest even: ia holds the shifted-out
             * bits, MSB-aligned; 0x80000000 is exactly half an
             * ULP of the result.
             */
            if ( (ia > 0x80000000) ||
                 ((ia == 0x80000000) && (ir & 1)) ) {
                ir++;
            }
        }
    }
    return ir;
}

int
main()
{
    cudaError_t status;
    unsigned short *convertedFloats = 0;
    unsigned short *deviceFloats = 0;
    int cDifferent = 0;
    long long cTotalDifferent = 0;

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
        cTotalDifferent += cDifferent;
    }
    if ( cTotalDifferent ) {
        printf( "Done: FAILED (%lld mismatches)\n", cTotalDifferent );
        return 1;
    }
    printf( "Done: success\n" );
    return 0;
Error:
    printf( "Error\n" );
    return 1;
}


