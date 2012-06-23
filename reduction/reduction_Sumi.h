/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

struct CReduction_Sumi {
public:
    CReduction_Sumi();
    int sum;

    CReduction_Sumi& operator +=( int a );
    volatile CReduction_Sumi& operator +=( int a ) volatile;

    CReduction_Sumi& operator +=( const CReduction_Sumi& a );
    volatile CReduction_Sumi& operator +=( volatile CReduction_Sumi& a ) volatile;

};

inline __device__ __host__
CReduction_Sumi::CReduction_Sumi()
{
    sum = 0;
}

inline __device__ __host__
CReduction_Sumi&
CReduction_Sumi::operator +=( int a )
{
    sum += a;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumi&
CReduction_Sumi::operator +=( int a ) volatile
{
    sum += a;
    return *this;
}

inline __device__ __host__
CReduction_Sumi&
CReduction_Sumi::operator +=( const CReduction_Sumi& a )
{
    sum += a.sum;
    return *this;
}

inline __device__ __host__
volatile CReduction_Sumi&
CReduction_Sumi::operator +=( volatile CReduction_Sumi& a ) volatile
{
    sum += a.sum;
    return *this;
}

inline int
operator!=( const CReduction_Sumi& a, const CReduction_Sumi& b )
{
    return a.sum != b.sum;
}
