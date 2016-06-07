/*
 *
 * chNUMA.h
 *
 * Header that wraps NUMA allocation/free functions for
 * Linux and Windows.
 *
 * Copyright (c) 2016, Archaea Software, LLC.
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

#ifdef WINDOWS

size_t
chNUMAgetPageSize()
{
    SYSTEM_INFO system_info;
    GetSystemInfo (&system_info);
    return (size_t) system_info.dwPageSize;
}

bool
chNUMAnumNodes( int *p )
{
    ULONG maxNode;
    if ( GetNumaHighestNodeNumber( &maxNode ) ) {
        *p = (int) maxNode+1;
        return true;
    }
    return false;
}

void *
chNUMApageAlignedAlloc( size_t bytes, int node )
{
    void *ret;
    ret = VirtualAllocExNuma( GetCurrentProcess(),
                              NULL,
                              bytes,
                              MEM_COMMIT | MEM_RESERVE,
                              PAGE_READWRITE,
                              node );
    return ret;
}

void
chNUMApageAlignedFree( void *p )
{
    VirtualFreeEx( GetCurrentProcess(), p, 0, MEM_RELEASE );
}

#else

#include <numa.h>
#include <unistd.h>

size_t
chNUMAgetPageSize()
{
  return (size_t) sysconf(_SC_PAGESIZE);
}

bool
chNUMAnumNodes( int *p )
{
    if ( numa_available() >= 0 ) {
        *p = numa_max_node() + 1;
        return true;
    }
    return false;
}

void *
chNUMApageAlignedAlloc( size_t bytes, int node )
{
    void *ret;
    ret = numa_alloc_onnode( bytes, node );
    return ret;
}

void
chNUMApageAlignedFree( void *p, size_t bytes )
{
    numa_free( p, bytes );
}

#endif

#include <stdint.h>

//
// Portable implementations that use the functions we just defined
//
bool
chNUMApageAlignedAllocHost( void **pp, size_t bytes, int node )
{
    bytes += chNUMAgetPageSize();
    void *p = chNUMApageAlignedAlloc( bytes, node );
    if ( NULL == p )
        goto Error;
    if ( cudaSuccess !=  cudaHostRegister( p, bytes, 0 ) )
        goto Error;
    *((size_t *) p) = bytes;
    *pp = (void *) ((char *) p+chNUMAgetPageSize());
    return true;
Error:
    if ( p ) {
        cudaHostUnregister( p );
        chNUMApageAlignedFree( p, bytes );
    }
    return false;
}

void
chNUMApageAlignedFreeHost( void *p )
{
    p = (void *) ((uintptr_t) p-chNUMAgetPageSize() );
    size_t bytes = *(size_t *) p;
    cudaHostUnregister( p );
    chNUMApageAlignedFree( p, bytes );
}
