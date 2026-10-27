/*
 *
 * chErrorNccl.h
 *
 * Error handling for NCCL:
 *     cu() and cuda() macros implement goto-based error
 *         error handling *, and
 *     chGetErrorString() maps a driver API error to a string.
 *
 * * The more-concise formulation of these macros is due to
 *   Allan MacKinnon.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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

#ifndef __CHERROR_NCCL_H__
#define __CHERROR_NCCL_H__

#ifdef DEBUG
#include <stdio.h>
#endif

#include <nccl.h>

//
// nccl() pastes the API prefix onto its parameter the same way cuda() and
// cu() do, so a call reads nccl(AllGather(...)) and expands to
// ncclAllGather(...), checks the result, and jumps to the function's Error
// label.
//
// It needs its own status variable rather than sharing the one cuda() uses,
// because ncclResult_t and cudaError_t are different types and a function
// that drives GPUs through NCCL almost always makes runtime API calls too.
// Declare both:
//
//     cudaError_t status;
//     ncclResult_t ncclStatus;
//

#ifdef DEBUG
#define nccl( fn ) do { \
        (ncclStatus) =  (nccl##fn); \
        if ( ncclSuccess != (ncclStatus) ) { \
            fprintf( stderr, "NCCL Failure (line %d of file %s):\n\t%s " \
                "returned %d (%s)\n", \
                __LINE__, __FILE__, #fn, ncclStatus, \
                ncclGetErrorString(ncclStatus) ); \
            goto Error; \
        } \
    } while (0);

#else

#define nccl( fn ) do { \
        (ncclStatus) =  (nccl##fn); \
        if ( ncclSuccess != (ncclStatus) ) { \
            goto Error; \
        } \
    } while (0);

#endif

#endif // __CHERROR_NCCL_H__
