/*
 *
 * chError.h
 *
 * Error handling for CUDA:
 *     cu() and cuda() macros implement goto-based error
 *         error handling *, and
 *     chGetErrorString() maps a driver API error to a string.
 *
 * * The more-concise formulation of these macros is due to
 *   Allan MacKinnon.
 *
 * Copyright (c) 2011-2022, Archaea Software, LLC.
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

#ifndef __CHERROR_H__
#define __CHERROR_H__

#ifdef DEBUG
#include <stdio.h>
#endif

#ifdef __HIPCC__
#include "chError_hip.h"
#else
#include "chError_cuda.h"
#endif

#if 0
//#else

template<typename T>
inline const char *
chGetErrorString( T status )
{
    return "CUDA support is not built in.";
}

//static inline const char* cudaGetErrorString( hipError_t error )
//{
//	return "CUDA support is not built in.";
//}

#define CUDART_CHECK( fn ) do { \
    status = (fn); \
    if ( hipSuccess != (status) ) { \
            goto Error; \
        } \
    } while (0);

#define CUDA_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            goto Error; \
        } \
    } while (0);

#endif

#endif // __CHERROR_H__

