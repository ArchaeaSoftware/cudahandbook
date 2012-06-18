/*
 *
 * chCommandLine.h
 *
 * CUDA handbook, dead simple command line parsing.
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

#ifndef __CUDAHANDBOOK_COMMANDLINE__
#define __CUDAHANDBOOK_COMMANDLINE__

#include <string.h>

static void
chCommandLinePassback( int *p, char *s )
{
    *p = atoi( s );
}

static void
chCommandLinePassback( char **p, char *s )
{
    *p = s;
}

//
// Passes back an integer or string.
//
template<class T>
static bool
chCommandLineGet( T *p, const char *keyword, int argc, char *argv[] )
{
    bool ret = false;
    for ( int i = 1; i < argc; i++ ) {
        char *s = argv[i];
        if ( *s == '-' ) {
            s++;
            if ( *s == '-' ) {
                s++;
            }
            if ( ! strcmp( s, keyword ) ) {
                if ( ++i <= argc ) {
                    chCommandLinePassback( p, argv[i] );
                    ret = true;
                }
            }
        }
    }
    return ret;
}

//
// Pass back true if the keyword is passed as a command line parameter
//
static bool
chCommandLineGetBool( const char *keyword, int argc, char *argv[] )
{
    bool ret = false;
    for ( int i = 1; i < argc; i++ ) {
        char *s = argv[i];
        if ( *s == '-' ) {
            s++;
            if ( *s == '-' ) {
                s++;
            }
            if ( ! strcmp( s, keyword ) ) {
                return true;
            }
        }
    }
    return ret;
}

#ifdef __CUDAHANDBOOK_SHMOO__
//
// if the user passes 'keyword,' set that as the range.
// otherwise, the user must specify minKeyword, maxKeyword, and 
// stepKeyword.  If one of those is missing, or if (max-min)%step!=0, 
// return failure.
//
// For example, if keyword is "Threads" and the parameters include 
// "Threads=N" pass back a chShmooRange with min and max equal to N.
// If keyword is "Cycles" and the parameters include
// "--minCycles 10 --maxCycles 20 --stepCycles 2"
// then pass back a corresponding shmoo.
//
// Using that example, if minCycles, maxCycles or stepCycles is 
// specified and not all three are specified, the function returns 
// false.
// 
//
template<>
static bool
chCommandLineGet( chShmooRange *range, const char *keyword, int argc, char *argv[] )
{
    // concatenate keyword onto the words min, max, step
    // require all three to be specified.
    int value, min, max, step;
    char s[256];

    if ( strlen(keyword) > 250 ) {
        return false;
    }
    strcpy( s, keyword );
    if ( chCommandLineGet( &value, s, argc, argv ) ) {
        range->Initialize( value );
        return true;
    }
    strcpy( s, "min" );
    strcat( s, keyword );
    if ( ! chCommandLineGet( &min, s, argc, argv) ) {
        return false;
    }
    strcpy( s, "max" );
    strcat( s, keyword );
    if ( ! chCommandLineGet( &max, s, argc, argv) ) {
        return false;
    }
    strcpy( s, "step" );
    strcat( s, keyword );
    if ( ! chCommandLineGet( &step, s, argc, argv ) ) {
        return false;
    }
    if ( ! range->Initialize( min, max, step ) ) {
        return false;
    }
    return true;
}
#endif // __CUDAHANDBOOK_SHMOO__

#endif
