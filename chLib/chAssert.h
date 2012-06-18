/*
 * chAssert.h
 *
 * CUDA handbook assertion defines.  
 * Portable to Windows, Linux, and Mac.
 * On Windows, triggers a breakpoint.
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

#include <stdio.h>
#ifndef _WIN32
#include <assert.h>
#endif

//
// These are basically a workaround for the way Microsoft Visual C++
// does not break in the debugger when an assertion fails.
//

#ifdef DEBUG

#ifdef _WIN32
#ifdef _WIN64
#define CH_ASSERT(predicate) if ( ! (predicate)) do { if(!(predicate)) { fprintf( stderr, "Assertion failed: %s at line %d in file %s\n", \
            #predicate, __LINE__, __FILE__ ); \
            DebugBreak(); \
        } \
    } while (0);
#else
#define CH_ASSERT(predicate) if ( ! (predicate)) do { if(!(predicate)) { fprintf( stderr, "Assertion failed: %s at line %d in file %s\n", \
            #predicate, __LINE__, __FILE__ ); \
            _asm int 3; \
        } \
    } while (0);
#endif

#else

// On non-Windows platforms, just use C runtime assert.
#define CH_ASSERT(predicate) assert(predicate)

#endif // not _WIN32

#else

#define CH_ASSERT(predicate)

#endif
