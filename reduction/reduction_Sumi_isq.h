/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
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
