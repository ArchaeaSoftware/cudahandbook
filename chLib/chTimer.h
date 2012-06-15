/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
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
