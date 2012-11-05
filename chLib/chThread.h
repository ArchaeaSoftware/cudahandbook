/*
 *
 * chThread.h
 *
 * Header file for helper classes and functions for threads.
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

#ifndef __CHTHREAD_H__
#define __CHTHREAD_H__

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>
#endif

#include <new>
#include <memory>
#include <vector>
#include <stdint.h>

namespace cudahandbook {

namespace threading {

#ifdef _WIN32
inline unsigned int
processorCount()
{
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );

    return sysinfo.dwNumberOfProcessors;
}

#else
inline unsigned int
processorCount()
{
    return sysconf( _SC_NPROCESSORS_ONLN );
}
#endif

#if 0
class mutex {
public:
    mutex() { }
    virtual ~mutex();

    // acquire and release mutex
    void acquire();
    void release();
    static mutex *mutexCreate( );
};
#endif

class workerThread
{
public:
    workerThread( int cpuThreadId = 0 ) { 
        m_cpuThreadId = cpuThreadId;
#ifdef _WIN32
        m_hThread = NULL;
        m_evWait = NULL;
        m_evDone = NULL;
#else
        memset( &m_semWait, 0, sizeof(m_semWait) );
        memset( &m_semDone, 0, sizeof(m_semDone) );
        memset( &m_thread, 0, sizeof(pthread_t) );
#endif
    }
    virtual ~workerThread() {
#ifdef _WIN32
        delegateSynchronous( NULL, NULL );
        WaitForSingleObject( m_hThread, INFINITE );
        CloseHandle( m_evWait );
        CloseHandle( m_evDone );
#else
        pthread_join( m_thread, NULL );
        sem_destroy( &m_semWait );
        sem_destroy( &m_semDone );
#endif
    }

    bool initialize( ) {
#ifdef _WIN32
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
#else
        pthread_attr_t attr;
        int ret = pthread_attr_init( &attr );
        if ( 0 != ret )
            goto Error;
        ret = sem_init( &m_semWait, 0, 0 );
        if ( 0 != ret )
            goto Error;
        ret = sem_init( &m_semDone, 0, 0 );
        if ( 0 != ret )
            goto Error;
        ret = pthread_create( &m_thread, &attr, threadRoutine, this );
        if ( 0 != ret )
            goto Error;
        return true;
    Error:
        sem_destroy( &m_semWait );
        sem_destroy( &m_semDone );
        return false;
#endif
    }

#ifdef _WIN32
    static DWORD __stdcall threadRoutine( LPVOID );
#else
    static void *threadRoutine( void * );
#endif

    //
    // call this from your app thread to delegate to the worker.
    // it will not return until your pointer-to-function has been called
    // with the given parameter.
    //
    bool delegateSynchronous( void (*pfn)(void *), void *parameter );

    //
    // call this from your app thread to delegate to the worker asynchronously.
    // Since it returns immediately, you must call waitAll later
    //

    bool delegateAsynchronous( void (*pfn)(void *), void *parameter );

    static bool waitAll( workerThread *p, size_t N );

private:
    unsigned int m_cpuThreadId;
#ifdef _WIN32
    HANDLE m_hThread;
    DWORD m_dwThreadId;
    HANDLE m_evWait;
    HANDLE m_evDone;
#else
    pthread_t m_thread;
    sem_t m_semWait;
    sem_t m_semDone;
#endif

    void (*m_pfnDelegatedWork)(void *);
    void *m_Parameter;
};

#ifdef _WIN32
DWORD __stdcall
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
#else
void *
workerThread::threadRoutine( void *_p )
{
    workerThread *p = (workerThread *) _p;
    bool bLoop = true;
    do {
        sem_wait( &p->m_semWait );
        if ( NULL == p->m_Parameter ) {
            bLoop = false;
        }
        else {
            (*p->m_pfnDelegatedWork)( p->m_Parameter );
            sem_post( &p->m_semDone );
        }
    } while ( bLoop );
    // fall through to exit if bLoop was set to false
    return 0;
}
#endif

inline bool
workerThread::delegateSynchronous( void (*pfn)(void *), void *parameter )
{
    m_pfnDelegatedWork = pfn;
    m_Parameter = parameter;
#ifdef _WIN32
    SetEvent( m_evWait );
    return WAIT_OBJECT_0 == WaitForSingleObject( m_evDone, INFINITE );
#else
    if ( 0 != sem_post( &m_semWait ) )
        return false;
    return 0 == sem_wait( &m_semDone );
#endif
}

inline bool
workerThread::delegateAsynchronous( void (*pfn)(void *), void *parameter )
{
    m_pfnDelegatedWork = pfn;
    m_Parameter = parameter;
#ifdef _WIN32
    SetEvent( m_evWait );
#else
    sem_post( &m_semWait );
#endif
    return true;
}

inline bool
workerThread::waitAll( workerThread *p, size_t N )
{
#ifdef _WIN32
    bool ret = false;
    HANDLE *pH = new HANDLE[N];
    if ( ! p ) {
        return false;
    }
    for ( size_t i = 0; i < N; i++ ) {
        pH[i] = p[i].m_evDone;
    }
    ret = WAIT_OBJECT_0 == WaitForMultipleObjects( N, pH, TRUE, INFINITE );
    delete[] pH;
    return ret;
#else
    for ( size_t i = 0; i < N; i++ ) {
        int ret = sem_wait( &p[i].m_semDone );
        if ( 0 != ret )
            return false;
    }
    return true;
#endif
}


//
// A worker thread pool is designed for coordinated dispatch onto the 
// same computation.  Each thread gets its own threadID and is expected 
// to be able to figure out its contribution to the computation based on 
// the thread ID and the thread count.
//
// workerThreadPool::delegateSynchronous spawns a multithreaded 
// computation in the worker threads, then waits until they have all
// finished before returning.
//
class workerThreadPool
{
public:
    workerThreadPool( int cThreads ) { 
    }
    virtual ~workerThreadPool() {

    }

    void delegateSynchronous( void (*pfn)(int tid, int numThreads, void *p), void *parameter )
    {
        
    }

private:
    size_t m_cThreads;
#ifdef _WIN32
    HANDLE *m_pThreads;

    // event pairs to 
    HANDLE *m_pWorkerWait;
    HANDLE *m_pWorkerSignal;
#else
#endif
};

} // namespace thread

} // namespace cudahandbook

#endif
