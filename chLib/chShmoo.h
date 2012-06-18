/*
 *
 * chShmoo.h
 *
 * CUDA handbook, C++ class to handle shmoo ranges.
 * This class serves two purposes: as a target for command line parsing
 * and as an iterator for programs that want to perform measurements 
 * over ranges of parameters (shmoos).
 *
 * If command line parsing of chShmooRange is desired, this header MUST
 * be included before chCommandLine.h.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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

#ifndef __CUDAHANDBOOK_SHMOO__
#define __CUDAHANDBOOK_SHMOO__

//
// C++ classes to assist in generating shmoo data
//

// Shmoos perform measurements over certain variables while holding
// others constant.  The chShmooRange class simplifies writing code
// to gather this data.

class chShmooRange {
public:
    chShmooRange( ) { }
    void Initialize( int value );
    bool Initialize( int min, int max, int step );
    bool isStatic() const { return m_min==m_max; }

    friend class chShmooIterator;

    int min() const { return m_min; }
    int max() const { return m_max; }

private:
    bool m_initialized;
    int m_min, m_max, m_step;
};

inline void
chShmooRange::Initialize( int value )
{
    m_min = m_max = value;
    m_step = 1;
    m_initialized = true;
}

inline bool
chShmooRange::Initialize( int min, int max, int step )
{
    if ( max < min )
        return false;
    if ( (max-min) % step )
        return false;
    m_min = min;
    m_max = max;
    m_step = step;
    m_initialized = true;
    return true;
}

class chShmooIterator
{
public:
    chShmooIterator( const chShmooRange& range );

    int operator *() const { return m_i; }
    operator bool() const { return m_i <= m_max; }
    void operator++(int) { m_i += m_step; };
private:
    int m_i;
    int m_max;
    int m_step;
};

inline
chShmooIterator::chShmooIterator( const chShmooRange& range )
{
    m_i = range.m_min;
    m_max = range.m_max;
    m_step = range.m_step;
}

#endif // __CUDAHANDBOOK_SHMOO__
