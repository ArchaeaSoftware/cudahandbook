/*
 *
 * chError_hip.h
 *
 * Error handling for HIP:
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

#ifndef __CHERROR_HIP_H__
#define __CHERROR_HIP_H__

#include <hip/hip_runtime.h>
#define cuda( fn ) do { \
	            status = (hip##fn); \
	            if ( hipSuccess != (status) ) { \
			                                    goto Error; \
			                                } \
	            } while (0);
#define CUDART_CHECK( fn ) do { \
    status = (fn); \
    if ( hipSuccess != (status) ) { \
	    goto Error; \
	} \
    } while (0);

typedef hipEvent_t cudaEvent_t;
typedef hipError_t cudaError_t;

#ifdef __cplusplus
template<typename T> hipError_t hipHostAlloc( T **pp, size_t N, unsigned int Flags ) {
    return hipHostMalloc( (void **) pp, N, Flags );
}

template<typename T> hipError_t hipHostGetDevicePointer( T **pp, void *p, unsigned int Flags ) {
    return hipHostGetDevicePointer( (void **) pp, p, Flags );
}
#endif

// entry points
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaFree hipFree
#define cudaHostFree hipHostFree
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaStreamDestroy hipStreamDestroy
#define cudaEventDestroy hipEventDestroy
#define cudaGetErrorString hipGetErrorString

// data types
typedef hipStream_t cudaStream_t;
typedef hipDeviceProp_t cudaDeviceProp;

// defines
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaHostAllocMapped 0
#define cudaHostAllocPortable 0

// error defines
#define cudaSuccess hipSuccess
#define cudaErrorUnknown hipErrorUnknown
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation

template<typename T>
inline const char *
chGetErrorString( T status )
{
    return hipGetErrorString(status);
}

// To use these macros, a local cudaError_t or CUresult called 'status' 
// and a label Error: must be defined.  In the debug build, the code will 
// emit an error to stderr.  In both debug and retail builds, the code will
// goto Error if there is an error.
//

#ifdef DEBUG
#define CUDART_CHECK( fn ) do { \
        (status) =  (fn); \
        if ( hipSuccess != (status) ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t" \
                "%s returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#define cuda( fn ) do { \
        (status) =  (hip##fn); \
        if ( hipSuccess != (status) ) { \
            fprintf( stderr, "HIP Runtime Failure (line %d of file %s):\n\t" \
                "%s returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, chGetErrorString(status) ); \
            goto Error; \
        } \
    } while (0);

#define cu( fn ) do { \
        (status) =  (hip##fn); \
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
    if ( hipSuccess != (status) ) { \
	    goto Error; \
	} \
    } while (0);

#define cuda( fn ) do { \
    status = (hip##fn); \
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

#define cu( fn ) do { \
        (status) =  (cu##fn); \
        if ( CUDA_SUCCESS != (status) ) { \
            goto Error; \
        } \
    } while (0);

#endif

#endif // __CHERROR_HIP_H__

