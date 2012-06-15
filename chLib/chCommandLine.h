/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

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
// Passes back an integer
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
