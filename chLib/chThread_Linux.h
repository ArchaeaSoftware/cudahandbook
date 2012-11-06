/*
 *
 * chThread_Linux.h
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

#ifndef __CHTHREAD_LINUX_H__
#define __CHTHREAD_LINUX_H__

#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>

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
    return sysconf( _SC_NPROCESSORS_ONLN );
}

class workerThread
{
public:
    workerThread( int cpuThreadId = 0 ) { 
        m_cpuThreadId = cpuThreadId;
        memset( &m_semWait, 0, sizeof(m_semWait) );
        memset( &m_semDone, 0, sizeof(m_semDone) );
        memset( &m_thread, 0, sizeof(pthread_t) );
    }
    virtual ~workerThread() {
        delegateSynchronous( NULL, NULL );
        pthread_join( m_thread, NULL );
        sem_destroy( &m_semWait );
        sem_destroy( &m_semDone );
    }

    bool initialize( ) {
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
    }

    static void *threadRoutine( void * );

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
    pthread_t m_thread;
    sem_t m_semWait;
    sem_t m_semDone;

    void (*m_pfnDelegatedWork)(void *);
    void *m_Parameter;
};

inline void *
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

inline bool
workerThread::delegateSynchronous( void (*pfn)(void *), void *parameter )
{
    m_pfnDelegatedWork = pfn;
    m_Parameter = parameter;
    if ( 0 != sem_post( &m_semWait ) )
        return false;
    return 0 == sem_wait( &m_semDone );
}

inline bool
workerThread::delegateAsynchronous( void (*pfn)(void *), void *parameter )
{
    m_pfnDelegatedWork = pfn;
    m_Parameter = parameter;
    return 0 == sem_post( &m_semWait );
}

inline bool
workerThread::waitAll( workerThread *p, size_t N )
{
    for ( size_t i = 0; i < N; i++ ) {
        int ret = sem_wait( &p[i].m_semDone );
        if ( 0 != ret )
            return false;
    }
    return true;
}

} // namespace threading

} // namespace cudahandbook

#endif
