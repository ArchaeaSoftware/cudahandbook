/*
 *
 * win32KernelStreamCallback.cu
 *
 * Microdemo to examine the behavior of stream callbacks on Win32.
 *
 * Periodically counts the number of CPU threads extant, and reports
 * that number.
 *
 * Build with: nvcc -I ../chLib <options> win32KernelStreamCallback.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
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

#include <stdio.h>

#include <chError.h>
#include <chTimer.h>

#include <tlhelp32.h>
#include <tchar.h>

__global__
void
NullKernel()
{
}

__global__
void
DereferenceNullKernel()
{
    *((volatile int *) 0) = 0xdeadbeef;
}

class CStreamCallbacksStats {
private:
    CRITICAL_SECTION m_cs;

    int m_cCallbacks;
    int m_intervalCount;
    int m_totalCallbacks;

    DWORD m_dwThreadId;

    HANDLE m_hEvent;      // event to signal
    int m_signalCount;    // number of callbacks to look for before signaling event
    int m_maxThreadCount;

    static const int intervalPeriod = 1000;


public:
    CStreamCallbacksStats() {
        InitializeCriticalSection( &m_cs );
        m_hEvent = NULL;
        m_signalCount = 0;
        m_intervalCount = intervalPeriod;
        m_cCallbacks = 0;
        m_totalCallbacks = 0;
        m_maxThreadCount = 0;
        m_dwThreadId = 0;
    }
    ~CStreamCallbacksStats() {
        DeleteCriticalSection( &m_cs );
        if ( m_hEvent ) {
            CloseHandle( m_hEvent );
        }
    }
    bool Initialize( int numCallbacksBeforeSignal ) {
        m_signalCount = numCallbacksBeforeSignal;
        m_hEvent = CreateEvent( NULL, FALSE, FALSE, NULL );
        return m_hEvent != NULL;
    }

    int getTotalCallbacks() const { return m_totalCallbacks; }

    static void CUDART_CB countCallbacks( cudaStream_t stream, cudaError_t status, void *userData );
    BOOL Wait() { return WaitForSingleObject( m_hEvent, INFINITE ); }

    int CountProcessThreads( );
} ;

int
CStreamCallbacksStats::CountProcessThreads( ) 
{
    DWORD dwOwnerPID = GetCurrentProcessId();
    int ret = 0;
    int numThreads = 0;
    HANDLE hThreadSnap = INVALID_HANDLE_VALUE; 
    THREADENTRY32 te32; 
 
    // Take a snapshot of all running threads  
    hThreadSnap = CreateToolhelp32Snapshot( TH32CS_SNAPTHREAD, 0 ); 
    if( hThreadSnap == INVALID_HANDLE_VALUE ) 
        return( FALSE ); 

    // Fill in the size of the structure before using it. 
    te32.dwSize = sizeof(THREADENTRY32 ); 
 
    // Retrieve information about the first thread,
    // and exit if unsuccessful
    if( !Thread32First( hThreadSnap, &te32 ) ) 
        goto Error;
    numThreads = 0;
    do {
        if( te32.th32OwnerProcessID == dwOwnerPID ) {
            numThreads += 1;
        }
    } while( Thread32Next(hThreadSnap, &te32 ) );

    EnterCriticalSection( &m_cs );
    if ( numThreads > m_maxThreadCount ) {
        m_maxThreadCount = numThreads;
    }
    LeaveCriticalSection( &m_cs );
    ret = numThreads;
Error:
    CloseHandle( hThreadSnap );
    return ret;
}


void CUDART_CB 
CStreamCallbacksStats::countCallbacks( cudaStream_t stream, cudaError_t status, void *userData )
{
    CStreamCallbacksStats *p = (CStreamCallbacksStats *) userData;
    const int intervalPeriod = 1000;

    EnterCriticalSection( &p->m_cs );
        DWORD dwThreadId = GetCurrentThreadId();
        if ( p->m_dwThreadId == 0 ) {
            printf( "Initializing thread ID\n" );
            p->m_dwThreadId = dwThreadId;
        }
        else {
            if ( p->m_dwThreadId != dwThreadId ) {
                printf( "Different thread ID\n" );
            }
        }
    LeaveCriticalSection( &p->m_cs );

    if ( 0 == InterlockedDecrement( (LPLONG) &p->m_intervalCount ) ) {
        InterlockedExchange( (LPLONG) &p->m_intervalCount, intervalPeriod );
        p->CountProcessThreads( );
    }

    if ( cudaSuccess != status ) {
        // confirm that kernel that faulted is reported properly
        printf( "status = %d\n", status );
        return;
    }
    if ( p->m_signalCount == InterlockedIncrement( (LONG *) &p->m_totalCallbacks ) ) {
        printf( "Signaling event\n" );
        SetEvent( p->m_hEvent );
    }
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    const int cIterations = 1000;

    CStreamCallbacksStats stats;

    stats.Initialize( cIterations );

    cuInit(0);//
    Sleep(1000);
    printf( "Max threads after cuInit(0): %d\n", stats.CountProcessThreads() );
    cudaFree(0);
    printf( "Max threads after cudaFree(0): %d\n", stats.CountProcessThreads() );
cuda(StreamAddCallback( NULL, CStreamCallbacksStats::countCallbacks, &stats, cudaStreamCallbackNonblocking ) );

    printf( "Max threads after cudaStreamAddCallback(): %d\n", stats.CountProcessThreads() );

    printf( "Measuring asynchronous launch time (with nonblocking callbacks)... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        NullKernel<<<1,1>>>();
        cuda(StreamAddCallback( NULL, CStreamCallbacksStats::countCallbacks, &stats, cudaStreamCallbackNonblocking ) );
        if ( i == 0 ) {
            printf( "Max threads: %d\n", stats.CountProcessThreads() );
        }
    }
    DereferenceNullKernel<<<1,1>>>();
    cuda(StreamAddCallback( NULL, CStreamCallbacksStats::countCallbacks, &stats, cudaStreamCallbackNonblocking ) );
    cudaDeviceSynchronize();
    chTimerGetTime( &stop );

    // race condition unless we wait here
    stats.Wait();

    printf( "%d callbacks\n", stats.getTotalCallbacks() );
    {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) cIterations;

        printf( "%.2f us\n", usPerLaunch );
    }

    return 0;
Error:
    printf( "CUDA error: %d (%s)\n", status, cudaGetErrorString( status ) );
    return 1;
}
