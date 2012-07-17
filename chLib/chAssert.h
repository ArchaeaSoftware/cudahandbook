/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

/*
 * chAssert.h
 *
 *    CUDA handbook assertion defines.
 *
 */

#include <stdio.h>
#ifndef _WIN32
#include <assert.h>
#endif

//
// These are basically a workaround for the way Microsoft Visual C++
// does not break in the debugger when an assertion fails.
//

#if 1//def DEBUG

#ifdef _WIN32
#ifdef _WIN64
#define CH_ASSERT(predicate) if ( ! (predicate)) __debugbreak();
#else
#define CH_ASSERT(predicate) if ( ! (predicate)) _asm int 3
#endif

#if 0
do { if(!(predicate)) { fprintf( stderr, "Assertion failed: %s at line %d in file %s\n", \
            #predicate, __LINE__, __FILE__ ); \
            _asm int 3; \
        } \
    } while (0);
#endif

#else

// just use C runtime assert on non-Windows platforms.
#define CH_ASSERT(predicate) assert(predicate)

#endif // not _WIN32

#else

#define CH_ASSERT(predicate)

#endif
