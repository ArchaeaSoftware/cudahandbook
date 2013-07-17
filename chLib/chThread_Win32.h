/*
 *
 * chThread_Win32.h
 *
 * Win32 implementation of helper classes and functions for threads.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

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

#ifndef __CHTHREAD_WIN32_H__
#define __CHTHREAD_WIN32_H__

#include <windows.h>

#include <string.h>

#include <new>
#include <memory>
#include <vector>
#include <stdint.h>

namespace cudahandbook {

namespace threading {

inline unsigned int
processorCount()
{
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );

    return sysinfo.dwNumberOfProcessors;
}

class workerThread
{
public:
    workerThread( int cpuThreadId = 0 ) { 
        m_cpuThreadId = cpuThreadId;
        m_hThread = NULL;
        m_evWait = NULL;
        m_evDone = NULL;
    }
    virtual ~workerThread() {
        delegateSynchronous( NULL, NULL );
        WaitForSingleObject( m_hThread, INFINITE );
        CloseHandle( m_evWait );
        CloseHandle( m_evDone );
    }

    bool initialize( ) {
        m_evWait = CreateEvent( NULL, FALSE, FALSE, NULL );
        if ( ! m_evWait )
            goto Error;
        m_evDone = CreateEvent( NULL, FALSE, FALSE, NULL );
        if ( ! m_evDone )
            goto Error;
        m_hThread = CreateThread( NULL, 0, (LPTHREAD_START_ROUTINE) threadRoutine, this, 0, &m_dwThreadId );
        if ( ! m_hThread )
            goto Error;
        return true;
    Error:
        if ( m_evWait ) {
            CloseHandle( m_evWait );
            m_evWait = NULL;
        }
        if ( m_evDone ) {
            CloseHandle( m_evDone );
            m_evDone = NULL;
        }
        return false;
    }

    static DWORD __stdcall threadRoutine( LPVOID );

    //
    // call this from your app thread to delegate to the worker.
    // it will not return until your pointer-to-function has been called
    // with the given parameter.
    //
    bool delegateSynchronous( void (*pfn)(void *), void *parameter );

    //
    // call from your app thread to delegate to the worker asynchronously.
    // Since it returns immediately, you must call waitAll() later.
    //

    bool delegateAsynchronous( void (*pfn)(void *), void *parameter );

    static bool waitAll( workerThread *p, size_t N );

private:
    unsigned int m_cpuThreadId;
    HANDLE m_hThread;
    DWORD m_dwThreadId;
    HANDLE m_evWait;
    HANDLE m_evDone;

    void (*m_pfnDelegatedWork)(void *);
    void *m_Parameter;
};

inline DWORD __stdcall
workerThread::threadRoutine( void *_p )
{
    workerThread *p = (workerThread *) _p;
    bool bLoop = true;
    do {
        WaitForSingleObject( p->m_evWait, INFINITE );
        if ( NULL == p->m_Parameter ) {
            bLoop = false;
        }
        else {
            (*p->m_pfnDelegatedWork)( p->m_Parameter );
            SetEvent( p->m_evDone );
        }
    } while ( bLoop );
    // fall through to exit if bLoop was set to false
    return 0;
}

inline bool
workerThread::delegateSynchronous( void (*pfn)(void *), void *parameter )
{
    m_pfnDelegatedWork = pfn;
    m_Parameter = parameter;
    SetEvent( m_evWait );
    return WAIT_OBJECT_0 == WaitForSingleObject( m_evDone, INFINITE );
}

inline bool
workerThread::delegateAsynchronous( void (*pfn)(void *), void *parameter )
{
    m_pfnDelegatedWork = pfn;
    m_Parameter = parameter;
    SetEvent( m_evWait );
    return true;
}

inline bool
workerThread::waitAll( workerThread *p, size_t N )
{
    bool ret = false;
    HANDLE *pH = new HANDLE[N];
    if ( ! p ) {
        delete[] pH;
        return false;
    }
    for ( size_t i = 0; i < N; i++ ) {
        pH[i] = p[i].m_evDone;
    }
    ret = WAIT_OBJECT_0 == WaitForMultipleObjects( N, pH, TRUE, INFINITE );
    delete[] pH;
    return ret;
}

} // namespace threading

} // namespace cudahandbook

#endif
