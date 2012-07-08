/*
 *
 * Header for CReduction_Sumi_isq class, which may be used in conjunction
 * with templated reduction formulations to compute the sum and the sum
 * of squares of an array of integers in one pass.
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

struct CReduction_Sumi_isq {
public:
    CReduction_Sumi_isq();
    int sum;
    long long sumsq;

    CReduction_Sumi_isq& operator +=( int a );
    volatile CReduction_Sumi_isq& operator +=( int a ) volatile;

    CReduction_Sumi_isq& operator +=( const CReduction_Sumi_isq& a );
    volatile CReduction_Sumi_isq& operator +=( volatile CReduction_Sumi_isq& a ) volatile;

};

inline __device__ __host__
CReduction_Sumi_isq::CReduction_Sumi_isq()
{
    sum = 0;
    sumsq = 0;
}

inline __device__ __host__
CReduction_Sumi_isq&
CReduction_Sumi_isq::operator +=( int a )
{
    sum += a;
    sumsq += (long long) a*a;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumi_isq&
CReduction_Sumi_isq::operator +=( int a ) volatile
{
    sum += a;
    sumsq += (long long) a*a;
    return *this;
}

inline __device__ __host__
CReduction_Sumi_isq&
CReduction_Sumi_isq::operator +=( const CReduction_Sumi_isq& a )
{
    sum += a.sum;
    sumsq += a.sumsq;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumi_isq&
CReduction_Sumi_isq::operator +=( volatile CReduction_Sumi_isq& a ) volatile
{
    sum += a.sum;
    sumsq += a.sumsq;
    return *this;
}

inline int
operator!=( const CReduction_Sumi_isq& a, const CReduction_Sumi_isq& b )
{
    return a.sum != b.sum && a.sumsq != b.sumsq;
}


//
// from Reduction SDK sample:
// specialize to avoid unaligned memory 
// access compile errors
//
template<>
struct SharedMemory<CReduction_Sumi_isq>
{
    __device__ inline operator       CReduction_Sumi_isq*()
    {
        extern __shared__ CReduction_Sumi_isq __smem_CReduction_Sumi_isq[];
        return (CReduction_Sumi_isq*)__smem_CReduction_Sumi_isq;
    }

    __device__ inline operator const CReduction_Sumi_isq*() const
    {
        extern __shared__ CReduction_Sumi_isq __smem_CReduction_Sumi_isq[];
        return (CReduction_Sumi_isq*)__smem_CReduction_Sumi_isq;
    }
};
