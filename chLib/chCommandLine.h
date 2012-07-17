/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
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
// Passes back an integer or string
//
template<typename T>
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
// otherwise, the user must specify minKeyword, maxKeyword, and stepKeyword
// if one of those is missing, or if (max-min)%step!=0, return failure.
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
    if ( ! chCommandLineGet( &min, s, argc, argv ) ) {
        return false;
    }
    strcpy( s, "max" );
    strcat( s, keyword );
    if ( ! chCommandLineGet( &max, s, argc, argv ) ) {
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
