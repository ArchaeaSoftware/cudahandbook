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
 * Copyright (c) 2011-2016, Archaea Software, LLC.
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

#include <chCUDA.h>

#ifndef NO_CUDA

template<typename T>
inline const char *
chGetErrorString( T status )
{
    return cudaGetErrorString(status);
}

template<>
inline const char *
chGetErrorString( CUresult status )
{
    switch ( status ) {
#define ErrorValue(Define) case Define: return #Define;
        ErrorValue(CUDA_SUCCESS)
        ErrorValue(CUDA_ERROR_INVALID_VALUE)
        ErrorValue(CUDA_ERROR_OUT_OF_MEMORY)
        ErrorValue(CUDA_ERROR_NOT_INITIALIZED)
        ErrorValue(CUDA_ERROR_DEINITIALIZED)
        ErrorValue(CUDA_ERROR_PROFILER_DISABLED)
        ErrorValue(CUDA_ERROR_PROFILER_NOT_INITIALIZED)
        ErrorValue(CUDA_ERROR_PROFILER_ALREADY_STARTED)
        ErrorValue(CUDA_ERROR_PROFILER_ALREADY_STOPPED)
        ErrorValue(CUDA_ERROR_NO_DEVICE)
        ErrorValue(CUDA_ERROR_INVALID_DEVICE)
        ErrorValue(CUDA_ERROR_INVALID_IMAGE)
        ErrorValue(CUDA_ERROR_INVALID_CONTEXT)
        ErrorValue(CUDA_ERROR_CONTEXT_ALREADY_CURRENT)
        ErrorValue(CUDA_ERROR_MAP_FAILED)
        ErrorValue(CUDA_ERROR_UNMAP_FAILED)
        ErrorValue(CUDA_ERROR_ARRAY_IS_MAPPED)
        ErrorValue(CUDA_ERROR_ALREADY_MAPPED)
        ErrorValue(CUDA_ERROR_NO_BINARY_FOR_GPU)
        ErrorValue(CUDA_ERROR_ALREADY_ACQUIRED)
        ErrorValue(CUDA_ERROR_NOT_MAPPED)
        ErrorValue(CUDA_ERROR_NOT_MAPPED_AS_ARRAY)
        ErrorValue(CUDA_ERROR_NOT_MAPPED_AS_POINTER)
        ErrorValue(CUDA_ERROR_ECC_UNCORRECTABLE)
        ErrorValue(CUDA_ERROR_UNSUPPORTED_LIMIT)
        ErrorValue(CUDA_ERROR_CONTEXT_ALREADY_IN_USE)
        ErrorValue(CUDA_ERROR_INVALID_SOURCE)
        ErrorValue(CUDA_ERROR_FILE_NOT_FOUND)
        ErrorValue(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND)
        ErrorValue(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED)
        ErrorValue(CUDA_ERROR_OPERATING_SYSTEM)
        ErrorValue(CUDA_ERROR_INVALID_HANDLE)
        ErrorValue(CUDA_ERROR_NOT_FOUND)
        ErrorValue(CUDA_ERROR_NOT_READY)
        ErrorValue(CUDA_ERROR_LAUNCH_FAILED)
        ErrorValue(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
        ErrorValue(CUDA_ERROR_LAUNCH_TIMEOUT)
        ErrorValue(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING)
        ErrorValue(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
        ErrorValue(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
        ErrorValue(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE)
        ErrorValue(CUDA_ERROR_CONTEXT_IS_DESTROYED)
#if CUDA_VERSION >= 4010
        ErrorValue(CUDA_ERROR_ASSERT)
        ErrorValue(CUDA_ERROR_TOO_MANY_PEERS)
        ErrorValue(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
        ErrorValue(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
#endif
        ErrorValue(CUDA_ERROR_STUB_LIBRARY)
        ErrorValue(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)
        ErrorValue(CUDA_ERROR_DEVICE_NOT_LICENSED)
        ErrorValue(CUDA_ERROR_INVALID_PTX)
        ErrorValue(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT)
        ErrorValue(CUDA_ERROR_NVLINK_UNCORRECTABLE)
        ErrorValue(CUDA_ERROR_JIT_COMPILER_NOT_FOUND)
        ErrorValue(CUDA_ERROR_JIT_COMPILATION_DISABLED)
        ErrorValue(CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
        ErrorValue(CUDA_ERROR_ILLEGAL_STATE)
        ErrorValue(CUDA_ERROR_ILLEGAL_ADDRESS)
        ErrorValue(CUDA_ERROR_HARDWARE_STACK_ERROR)
        ErrorValue(CUDA_ERROR_ILLEGAL_INSTRUCTION)
        ErrorValue(CUDA_ERROR_MISALIGNED_ADDRESS)
        ErrorValue(CUDA_ERROR_INVALID_ADDRESS_SPACE)
        ErrorValue(CUDA_ERROR_INVALID_PC)
        ErrorValue(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE)
        ErrorValue(CUDA_ERROR_NOT_PERMITTED)
        ErrorValue(CUDA_ERROR_NOT_SUPPORTED)
        ErrorValue(CUDA_ERROR_SYSTEM_NOT_READY)
        ErrorValue(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH)
        ErrorValue(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_MERGE)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_UNMATCHED)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_UNJOINED)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_ISOLATION)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT)
        ErrorValue(CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD)
        ErrorValue(CUDA_ERROR_TIMEOUT)
        ErrorValue(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE)
        ErrorValue(CUDA_ERROR_CAPTURED_EVENT)

        ErrorValue(CUDA_ERROR_UNKNOWN)
    }
    return "chGetErrorString - unknown error value";
}



//
// To use these macros, a local cudaError_t or CUresult called 'status' 
// and a label Error: must be defined.  In the debug build, the code will 
// emit an error to stderr.  In both debug and retail builds, the code will
// goto Error if there is an error.
//

#ifdef DEBUG
#define CUDART_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( cudaSuccess != (status) ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t" \
                "%s returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#define cuda( fn ) do { \
        (status) =  (cuda##fn); \
        if ( cudaSuccess != (status) ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t" \
                "%s returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#define cu( fn ) do { \
        (status) =  (cu##fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t%s "\
                "returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#define CUDA_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t%s "\
                "returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#else


#define CUDART_CHECK( fn ) do { \
    status = (fn); \
    if ( cudaSuccess != (status) ) { \
	    goto Error; \
	} \
    } while (0);

#define cuda( fn ) do { \
    status = (cuda##fn); \
    if ( cudaSuccess != (status) ) { \
	    goto Error; \
	} \
    } while (0);


#define CUDA_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            goto Error; \
        } \
    } while (0);

#define cu( fn ) do { \
        (status) =  (cu##fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            goto Error; \
        } \
    } while (0);

#endif

#else

template<typename T>
inline const char *
chGetErrorString( T status )
{
    return "CUDA support is not built in.";
}

static inline const char* cudaGetErrorString( cudaError_t error )
{
	return "CUDA support is not built in.";
}

#define CUDART_CHECK( fn ) do { \
    status = (fn); \
    if ( cudaSuccess != (status) ) { \
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

#endif
