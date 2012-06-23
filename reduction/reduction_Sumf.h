/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

struct CReduction_Sumf {
public:
    CReduction_Sumf();
    float sum;

    CReduction_Sumf& operator +=( float a );
    volatile CReduction_Sumf& operator +=( float a ) volatile;

    CReduction_Sumf& operator +=( const CReduction_Sumf& a );
    volatile CReduction_Sumf& operator +=( volatile CReduction_Sumf& a ) volatile;

};

inline __device__ __host__
CReduction_Sumf::CReduction_Sumf()
{
    sum = 0.0f;
}

inline __device__ __host__
CReduction_Sumf&
CReduction_Sumf::operator +=( float a )
{
    sum += a;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumf&
CReduction_Sumf::operator +=( float a ) volatile
{
    sum += a;
    return *this;
}

inline __device__ __host__
CReduction_Sumf&
CReduction_Sumf::operator +=( const CReduction_Sumf& a )
{
    sum += a.sum;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumf&
CReduction_Sumf::operator +=( volatile CReduction_Sumf& a ) volatile
{
    sum += a.sum;
    return *this;
}

inline int
operator!=( const CReduction_Sumf& a, const CReduction_Sumf& b )
{
    return fabsf( a.sum - b.sum ) > 1e-5f;
}
