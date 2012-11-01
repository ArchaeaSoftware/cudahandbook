/*
 *
 * chThread.h
 *
 * Header file for helper classes and functions for threads.
 * Implementations are given in chThread_win32.h, chThread_posix.h
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
#include <new>
#endif

namespace cudahandbook {

namespace thread {

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
    return sysconf( _SC_NPROCESSORS_ONLN );}
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

#ifdef _WIN32
class mutex {
public:
    mutex() { InitializeCriticalSection( &m_cs ); }
    ~mutex() { DeleteCriticalSection( &m_cs ); }
    void acquire() { EnterCriticalSection( &m_cs ); }
    void release() { LeaveCriticalSection( &m_cs ); }
private:
    CRITICAL_SECTION m_cs;
};

#else
#include <pthread.h>

class mutex
{
public:
    mutex() { pthread_init_mutex( &m_mutex, NULL ); }
    ~mutex() { pthread_mutex_destroy( &m_mutex ); }
    void acquire() { pthread_mutex_lock( &m_mutex ); }
    void release() { pthread_mutex_unlock( &m_mutex ); }
private:
    pthread_mutex_t m_mutex;
};

#endif

// RAII (Resource Acquisition Is Initialization) idiom to automatically
// release a mutex when it drops out of scope.
class mutex_acquire
{
public:
    mutex_acquire( mutex *m ) {
        m_mutex = m;
        m->acquire();
    }
    ~mutex_acquire() { m_mutex->release(); }
private:
    mutex *m_mutex;
};


#ifdef _WIN32

template<typename T, typename ExceptionType>
class exception {
public:
    exception( ExceptionType code ) { m_code = code; }
    ExceptionType getCode() const {
        return m_code;
    }
private:
    ExceptionType m_code;

};

class semaphore
{
public:
    static const int max_sem_count = 0x7fffffff;

    semaphore() { 
        m_hSemaphore = CreateSemaphore( NULL, 0, max_sem_count, NULL );
        if ( ! m_hSemaphore ) {
            throw exception<semaphore, DWORD>( GetLastError() );
        }
    }
    ~semaphore() { 
        if ( m_hSemaphore ) {
            CloseHandle( m_hSemaphore );
        }
    }

    //
    // caution - other operating systems do not support handle duplication,
    // so this implementation does not use it.
    //
    // As a result, all copies of a cudahandbook::semaphore are invalidated 
    // when any instance executes the destructor.
    //
    // If this is a problem, use operator new and shared_ptr<> to reference
    // count the object.  Or if you are Windows only, uncomment the copy
    // constructor and assignment operator given below.
    //
#if 1
    semaphore( const semaphore& x ) { 
        DuplicateHandle( GetCurrentProcess(), x.m_hSemaphore, GetCurrentProcess(), &m_hSemaphore, 0, TRUE, DUPLICATE_SAME_ACCESS );
    }

    semaphore& operator=( const semaphore& x ) {
        if ( m_hSemaphore ) CloseHandle( m_hSemaphore );
        DuplicateHandle( GetCurrentProcess(), x.m_hSemaphore, GetCurrentProcess(), &m_hSemaphore, 0, TRUE, DUPLICATE_SAME_ACCESS );
        return *this;
    }
#endif

    //
    // wait for all semaphores to be signaled before returning
    //
    static bool waitSemaphores( semaphore *p, size_t N ) {
#if 0
        bool ret = false;
        HANDLE *handles = new HANDLE[N];
        if ( handles ) {
            for ( size_t i = 0; i < N; i++ ) {
                handles[i] = p[i].m_hSemaphore;
            }
            if ( WAIT_OBJECT_0 == WaitForMultipleObjects( (DWORD) N, (HANDLE *) handles, TRUE, INFINITE ) ) {
                ret = true;
            }
            delete[] handles;
        }
#endif
        if ( WAIT_OBJECT_0 == WaitForMultipleObjects( (DWORD) N, (HANDLE *) p, TRUE, INFINITE ) ) {
            return true;
        }
        return false;
    }

    // signal semaphore
    void signal() { ReleaseSemaphore( m_hSemaphore, 1, NULL ); }

    // wait for semaphore to be signaled
    void wait() { WaitForSingleObject( m_hSemaphore, INFINITE ); }

private:
    HANDLE m_hSemaphore;
};

class thread
{
public:
    thread( void *pfnStart, void *parameter ) {
        m_Thread = CreateThread( NULL, 0, (LPTHREAD_START_ROUTINE) pfnStart, parameter, 0, &m_ThreadId );
        if ( ! m_Thread ) {
            throw exception<thread, DWORD>( GetLastError() );
        }
    }

    // wait until all given threads have exited
    static void joinThreads( thread **, size_t );
private:
    HANDLE m_Thread;
    DWORD m_ThreadId;
};


class workerThread : public thread
{
public:
    workerThread( int cpuThreadId = 0 ) : thread(startRoutine, this) 
        { m_cpuThreadId = cpuThreadId; }
    void setID( int id ) { m_cpuThreadId = id; }
    virtual ~workerThread();

    static DWORD __stdcall startRoutine( LPVOID );

    //
    // call this from your app thread to delegate to the worker.
    // it will not return until your pointer-to-function has been called
    // with the given parameter.
    //
    bool delegateSynchronous( void (*pfn)(void *), void *parameter );

    //
    // call delegateAsynchronous from your app thread to delegate to a worker.
    // Because it returns before the delegated work has been done, you must
    // then call waitAll to synchronize.
    //
    bool delegateAsynchronous( void (*pfn)(void *), void *parameter );
    static bool waitAll( workerThread *p, size_t N );

private:
    unsigned int m_cpuThreadId;
    semaphore m_semWait;
    semaphore m_semDone;

    void (*m_pfnDelegatedWork)(void *);
    void *m_Parameter;
};

inline
workerThread::~workerThread()
{
    // tell worker thread to exit
    delegateSynchronous( NULL, NULL );
}

DWORD __stdcall
workerThread::startRoutine( LPVOID _p )
{
    workerThread *p = (workerThread *) _p;
    bool bLoop = true;
    do {
        p->m_semWait.wait();
        if ( NULL == p->m_Parameter ) {
            bLoop = false;
        }
        else {
            (*p->m_pfnDelegatedWork)( p->m_Parameter );
            p->m_semDone.signal();
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
    m_semWait.signal();
    m_semDone.wait();
    return true;
}

inline bool
workerThread::delegateAsynchronous( void (*pfn)(void *), void *parameter )
{
    m_pfnDelegatedWork = pfn;
    m_Parameter = parameter;
    m_semWait.signal();
    return true;
}

inline bool
workerThread::waitAll( workerThread *p, size_t N )
{
    semaphore *pSemaphores = new semaphore[N];
    if ( pSemaphores ) {
        for ( size_t i = 0; i < N; i++ ) {
            pSemaphores[i] = p[i].m_semDone;
        }
        semaphore::waitSemaphores( pSemaphores, N );
        delete[] pSemaphores;
        return true;
    }
    return false;
}

#endif


//
// A squad of threads is designed for coordinated dispatch onto the same
// computation.  Each thread gets its own threadID and is expected to be
// able to figure out its contribution to the computation based on the
// thread ID and the thread count.  threadsquad::delegateSynchronous
// spawns a multithreaded computation in the worker threads, then waits
// until they have all finished before returning.
//
class threadsquad
{
public:
    threadsquad( int cThreads ) { 
        try {
            m_pWorkerWait = new semaphore[m_cThreads];
            m_pWorkerSignal = new semaphore[m_cThreads];
            m_pThreads = new workerThread[cThreads];
        }
        catch( std::bad_alloc e ) {
            delete[] m_pWorkerWait;
            delete[] m_pWorkerSignal;
            delete[] m_pThreads;
        }
        m_cThreads = cThreads;
        for ( int i = 0; i < cThreads; i++ ) {
            m_pThreads[i].setID( i );
        }
    }
    virtual ~threadsquad() { if ( m_pThreads ) delete[] m_pThreads; }

    void delegateSynchronous( void (*pfn)(int tid, int numThreads, void *p), void *parameter )
    {
        
    }

private:
    workerThread *m_pThreads;
    size_t m_cThreads;

    semaphore *m_pWorkerWait;
    semaphore *m_pWorkerSignal;
};

} // namespace thread

} // namespace cudahandbook

#endif
