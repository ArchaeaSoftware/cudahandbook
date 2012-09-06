/*
 *
 * solveQuadratic.cu
 *
 * Microdemo to illustrate conditional code.
 *
 * Build with: nvcc [--gpu-architecture sm_xx] [-D USE_FLOAT] [-D USE_IF_STATEMENT] --cubin solveQuadratic.cu
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

#ifdef USE_IF_STATEMENT
template<class Real>
__device__ void
solveQuadratic( Real& x0, Real& x1, Real a, Real b, Real c )
{
    Real Determinant = __fsqrt_rn( b*b - 4.0f*a*c );
    if ( b >= 0.0f ) {
        x0 = (-b - Determinant) / (2.0f*a);
        x1 = (2.0f*c) / (-b - Determinant);
    }
    else {
        x0 = (-b + Determinant) / (2.0f*a);
        x1 = (2.0f*c) / (-b + Determinant);
    }
}
#else
template<class Real>
__device__ void
solveQuadratic( Real& x0, Real& x1, Real a, Real b, Real c )
{
    Real Determinant = __fsqrt_rn( b*b - 4.0f*a*c );
    Real adjDeterminant = (b < 0.0f) ? -b+Determinant : -b-Determinant;
    x0 = adjDeterminant / (2.0f*a);
    x1 = (2.0f*c) / adjDeterminant;
}
#endif


template<class Real>
__global__ void
solveQuadratics( 
    Real *x0, Real *x1, 
    const Real *pA, 
    const Real *pB, 
    const Real *pC, 
    size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x+threadIdx.x; 
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        solveQuadratic( x0[i], x1[i], pA[i], pB[i], pC[i] );
    }
}


int
main()
{
#ifndef USE_FLOAT
    solveQuadratics<double><<<2, 384>>>( NULL, NULL, NULL, NULL, NULL, 0 );
#else
    solveQuadratics<float><<<2, 384>>>( NULL, NULL, NULL, NULL, NULL, 0 );
#endif    
    return 0;
}
