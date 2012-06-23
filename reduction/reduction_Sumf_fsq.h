/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

struct CReduction_Sumf_fsq {
public:
    CReduction_Sumf_fsq();
    float sum;
    float sumsq;

    CReduction_Sumf_fsq& operator +=( float a );
    volatile CReduction_Sumf_fsq& operator +=( float a ) volatile;

    CReduction_Sumf_fsq& operator +=( const CReduction_Sumf_fsq& a );
    volatile CReduction_Sumf_fsq& operator +=( volatile CReduction_Sumf_fsq& a ) volatile;

};

inline __device__ __host__
CReduction_Sumf_fsq::CReduction_Sumf_fsq()
{
    sum = 0;
    sumsq = 0;
}

inline __device__ __host__
CReduction_Sumf_fsq&
CReduction_Sumf_fsq::operator +=( float a )
{
    sum += a;
    sumsq += a*a;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumf_fsq&
CReduction_Sumf_fsq::operator +=( float a ) volatile
{
    sum += a;
    sumsq += a*a;
    return *this;
}

inline __device__ __host__
CReduction_Sumf_fsq&
CReduction_Sumf_fsq::operator +=( const CReduction_Sumf_fsq& a )
{
    sum += a.sum;
    sumsq += a.sumsq;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumf_fsq&
CReduction_Sumf_fsq::operator +=( volatile CReduction_Sumf_fsq& a ) volatile
{
    sum += a.sum;
    sumsq += a.sumsq;
    return *this;
}

inline int
operator!=( const CReduction_Sumf_fsq& a, const CReduction_Sumf_fsq& b )
{
    return a.sum != b.sum && a.sumsq != b.sumsq;
}
