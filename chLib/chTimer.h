/*
 *
 * chTimer.h
 *
 * Portable, header-only timing library for the CUDA handbook.
 * All of the functions are inline, so no compiling or linking
 * is needed.  QueryPerformanceCounter() is used on Windows, and
 * gettimeofday() is used on other platforms.
 *
 *     chTimerTimestamp defines a type that can hold a timestamp.
 *     chTimerGetTime() writes the current time into a timestamp.
 *     chTimerElapsedTime() passes back the elapsed time between
 *         two timestamps.
 *     chTimerBandwidth() computes the bandwidth (bytes/s) given
 *         two timestamps.
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

#ifndef __CUDAHANDBOOK_TIMER_H__
#define __CUDAHANDBOOK_TIMER_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32

#include <windows.h>

typedef LARGE_INTEGER chTimerTimestamp;

inline void
chTimerGetTime(chTimerTimestamp *p)
{
    QueryPerformanceCounter( p );
}

inline double
chTimerElapsedTime( chTimerTimestamp *pStart, chTimerTimestamp *pStop )
{
    double diff = (double) ( pStop->QuadPart - pStart->QuadPart );
    LARGE_INTEGER freq;
    QueryPerformanceFrequency( &freq );
    return diff / (double) freq.QuadPart;
}

#elif defined(__MACH__)

#include <stdint.h>
#include <mach/mach_time.h>

typedef uint64_t chTimerTimestamp;

inline void
chTimerGetTime( chTimerTimestamp *p )
{
    if (!p)
        return;
    *p = mach_absolute_time();
}

inline double
chTimerElapsedTime( chTimerTimestamp *pStart, chTimerTimestamp *pStop )
{
    mach_timebase_info_data_t timebaseInfo;
    static double conversion = -1.0;

    if (!pStart || !pStop)
        return 0.0;

    if (conversion < 0.0) {
        mach_timebase_info(&timebaseInfo);
        conversion = 1e-9 * (double)timebaseInfo.numer /
            (double)timebaseInfo.denom;
    }

    return conversion * (double)(*pStop - *pStart);
}

#else

#include <time.h>

// On Linux and MacOS, use clock_gettime()
typedef struct timespec chTimerTimestamp;

inline void
chTimerGetTime( chTimerTimestamp *p )
{
    clock_gettime(CLOCK_MONOTONIC, p);
}

inline double
chTimerElapsedTime( chTimerTimestamp *pStart, chTimerTimestamp *pStop )
{
    return (double) (pStop->tv_sec - pStart->tv_sec) + 
                    (pStop->tv_nsec - pStart->tv_nsec)/1e9;
}

#endif

inline double
chTimerBandwidth( chTimerTimestamp *pStart, chTimerTimestamp *pStop, double cBytes )
{
    double et = chTimerElapsedTime( pStart, pStop );
    return cBytes / et;
}

#ifdef __cplusplus
};
#endif

#endif
