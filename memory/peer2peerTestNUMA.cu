/*
 *
 * peer2peerTestNUMA.cu
 *
 * Explore NUMA properties of PCI Express bus hierarchy.
 *
 * Build with: nvcc -I ../chLib <options> peer2peerTestNUMA.cu
 * Requires: No minimum SM requirement.
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


#include <stdio.h>

#include <unistd.h>

#include <pthread.h>
#include <semaphore.h>

#include <assert.h>

#include <vector>
#include <list>

#include <iostream>
#include <fstream>

#include "chError.h"
#include "chTimer.h"
#include "chNUMA.h"

#define MAX_DEVICES 32

using namespace std;

void *g_hostBuffers[MAX_DEVICES];
// Indexed as follows: [device][event]
bool g_bEnabled[MAX_DEVICES][MAX_DEVICES];

// these are already defined on some platforms - make our
// own definitions that will work.
#undef min
#undef max
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((b)<(a)?(a):(b))

#define NUM_ARRAY_ELEMENTS(a) (sizeof(a)/sizeof((a)[0]))
#define CHECK_NONZERO(f) { int ret = (f); if ( 0 != ret) { fprintf(stderr, "%s returned %d (File %s at line %d)\n", #f, ret, __FILE__, __LINE__ ); exit(1); } }

// Grab this mutex before writing output, so it does not get garbled
pthread_mutex_t g_mutexOutput;

// Resources needed for each device
const size_t g_cBytes = 2560ULL*1048576;
size_t g_cIterations = 10;

class cEnumCPUGPU {
public:
    // make GPU by default
    cEnumCPUGPU( int i, bool bCPU ) { m_i = i; m_bCPU=bCPU; }

    friend cEnumCPUGPU makeGPU( int iGPU ) { return cEnumCPUGPU( iGPU, false ); }
    friend cEnumCPUGPU makeCPU( int iCPU ) { return cEnumCPUGPU( iCPU, true ); }

    bool bGPU() const { return !m_bCPU; }
    bool bCPU() const { return m_bCPU; }

    int getCPU() const {
        assert( m_bCPU );
        return m_i;
    }
    int getGPU() const {
        assert( !m_bCPU );
        return m_i;
    }

    friend ostream& operator<<( ostream&, const cEnumCPUGPU& );

private:
    bool m_bCPU;
    int m_i;
};

inline ostream&
operator<<( ostream& os, const cEnumCPUGPU& e )
{
    if ( e.m_bCPU ) {
        os << "cpu " << e.m_i;
    }
    else {
        os << "gpu " << e.m_i;
    }
    return os;
}

//
// Thread context structure
//
class CGPULoadDriver {
public:
    virtual ~CGPULoadDriver();

    bool TimeMemcpys();

    virtual bool PerformMemcpys() = 0;

    sem_t m_semWait;
    sem_t m_semDone;

    static void *ThreadProc( void * );

protected:
    CGPULoadDriver( cEnumCPUGPU dstDevice, cEnumCPUGPU srcDevice, cEnumCPUGPU eventDevice, size_t m_cBytes, bool bLatencyTest = false );

    size_t m_cBytes;

    cudaEvent_t m_evStart;
    cudaEvent_t m_evStop;

    cEnumCPUGPU m_dstDevice;
    cEnumCPUGPU m_srcDevice;
    cEnumCPUGPU m_eventDevice;

    bool m_bUseEvents;
    bool m_bLatencyTest;
};

class CGPUTestP2P : public CGPULoadDriver {
public:

    CGPUTestP2P( cEnumCPUGPU dst, cEnumCPUGPU src, size_t cBytes, bool bUseEvents, bool bLatencyTest = false );
    virtual ~CGPUTestP2P();
    virtual bool PerformMemcpys( );

    void *m_dptrDst;
    void *m_dptrSrc;
};

class CGPUTestP2PLatency : public CGPUTestP2P {
public:

    CGPUTestP2PLatency( cEnumCPUGPU dst, cEnumCPUGPU src, size_t cIterations, bool bUseEvents ) : CGPUTestP2P( dst, src, 4, bUseEvents, true ) { 
        cudaError_t status;
        m_cIterations = cIterations;
        cuda(Malloc( &m_dptrDstTimestamps, (cIterations+1)*sizeof(uint64_t) ) );
        cuda(Malloc( &m_dptrSrcTimestamps, (cIterations+1)*sizeof(uint64_t) ) );
        return;
    Error: 
        fprintf(stderr, "cudaMalloc failed\n" ); 
        exit(1);
    }
    virtual bool PerformMemcpys( );
private:
    size_t m_cIterations;
    uint64_t *m_dptrDstTimestamps, *m_dptrSrcTimestamps;
};


class CGPUTestH2D : public CGPULoadDriver {
public:

    CGPUTestH2D( cEnumCPUGPU dst, cEnumCPUGPU src, size_t cBytes, bool bUseEvents );
    ~CGPUTestH2D();
    virtual bool PerformMemcpys( );

    void *m_dptrDst;
    void *m_pSrc;
};


class CGPUTestD2H : public CGPULoadDriver {
public:

    CGPUTestD2H( cEnumCPUGPU dst, cEnumCPUGPU src, size_t cBytes, bool bUseEvents );
    ~CGPUTestD2H();
    virtual bool PerformMemcpys( );

    void *m_dptrSrc;
    void *m_pDst;
};

struct GPUPair {
    GPUPair( cEnumCPUGPU dst, cEnumCPUGPU src, bool bLatencyTest ): iDst(dst), iSrc(src), m_bLatencyTest(bLatencyTest) { }

    cEnumCPUGPU iDst, iSrc;
    bool m_bLatencyTest;

};

ostream&
operator<<( ostream& os, const GPUPair& p )
{
    os << p.iDst << " <- " << p.iSrc;
    return os;
}

CGPULoadDriver::CGPULoadDriver( cEnumCPUGPU dstDevice, cEnumCPUGPU srcDevice, cEnumCPUGPU eventDevice, size_t cBytes, bool bLatencyTest ):
    m_dstDevice( dstDevice ), m_srcDevice( srcDevice ), m_eventDevice( eventDevice ), m_bLatencyTest(bLatencyTest)
{
    cudaError_t status;

    m_evStart = m_evStop = 0;

    m_cBytes = cBytes;

    CHECK_NONZERO( sem_init( &m_semWait, 0, 1 ) );
    CHECK_NONZERO( sem_init( &m_semDone, 0, 1 ) );

    cuda(SetDevice( m_eventDevice.getGPU() ) );
    cuda(EventCreate( &m_evStart ) );
    cuda(EventCreate( &m_evStop ) );
    return;
Error:
    cout << "Error creating CGPULoadDriver( " << dstDevice << ", " << srcDevice << endl;
    exit(1);
}

CGPULoadDriver *
makeLoadDriver( cEnumCPUGPU dst, cEnumCPUGPU src, size_t bytes, bool bUseEvents , bool bLatencyTest)
{
    // CPU<->CPU is not valid
    assert( src.bGPU() || dst.bGPU() );
    if ( bLatencyTest ) {
        assert( src.bGPU() && dst.bGPU() );
        return new CGPUTestP2PLatency( dst, src, 100000, bUseEvents );
    }
    if ( dst.bGPU() && src.bGPU() ) {
        return new CGPUTestP2P( dst, src, bytes, bUseEvents );
    } else if ( dst.bGPU() && src.bCPU() ) {
        return new CGPUTestH2D( dst, src, bytes, bUseEvents );
    } else if ( dst.bCPU() && src.bGPU() ) {
        return new CGPUTestD2H( dst, src, bytes, bUseEvents );
    }
    return NULL;
}

CGPUTestP2P::CGPUTestP2P( cEnumCPUGPU dstDevice, cEnumCPUGPU srcDevice, size_t cBytes, bool bUseEvents, bool bLatencyTest ):
    CGPULoadDriver( dstDevice, srcDevice, srcDevice, cBytes, bLatencyTest )
{
    cudaError_t status;
    assert( dstDevice.bGPU() && srcDevice.bGPU() );
    m_bUseEvents = bUseEvents;
    cuda(SetDevice(m_dstDevice.getGPU() ) );
    cuda(Malloc( &m_dptrDst, g_cBytes ) );
    cuda(Memset( m_dptrDst, 0, g_cBytes ) );
    cuda(SetDevice(m_srcDevice.getGPU() ) );
    cuda(Malloc( &m_dptrSrc, g_cBytes ) );
    cuda(Memset( m_dptrSrc, 0, g_cBytes ) );
    return;
Error:
    cerr << "Error creating CGPUTestP2P " << dstDevice << ", " << srcDevice << endl;
    exit(1);
}

CGPUTestH2D::CGPUTestH2D( cEnumCPUGPU dstDevice, cEnumCPUGPU srcDevice, size_t cBytes, bool bUseEvents ):
    CGPULoadDriver( dstDevice, srcDevice, dstDevice, cBytes )
{
    cudaError_t status;
    assert( dstDevice.bGPU() && srcDevice.bCPU() );
    m_bUseEvents = bUseEvents;
    m_dptrDst = 0;
    m_pSrc = 0;
    cuda(SetDevice(m_dstDevice.getGPU() ) );
    cuda(Malloc( &m_dptrDst, m_cBytes ) );
    if ( ! chNUMApageAlignedAllocHost( &m_pSrc, m_cBytes, dstDevice.getGPU() ) )
        goto Error;
    return;
Error:
cerr << "Error creating CGPUTestH2D " << dstDevice << ", " << srcDevice << endl;
    exit(1);
}

CGPUTestD2H::CGPUTestD2H( cEnumCPUGPU dstDevice, cEnumCPUGPU srcDevice, size_t cBytes, bool bUseEvents ):
    CGPULoadDriver( dstDevice, srcDevice, srcDevice, cBytes )
{
    cudaError_t status;
    assert( dstDevice.bCPU() && srcDevice.bGPU() );
    m_bUseEvents = bUseEvents;
    m_pDst = 0;
    m_dptrSrc = 0;
    cuda(SetDevice(m_srcDevice.getGPU() ) );
    cuda(Malloc( &m_dptrSrc, g_cBytes ) );
    cuda(MallocHost( &m_pDst, g_cBytes ) );
    return;
Error:
    cerr << "Error creating CGPUTestD2H " << dstDevice << ", " << srcDevice << endl;
    exit(1);
}

CGPULoadDriver::~CGPULoadDriver()

{
    sem_destroy( &m_semWait );
    sem_destroy( &m_semDone );
    cudaSetDevice( m_eventDevice.getGPU() );
    cudaEventDestroy( m_evStart );
    cudaEventDestroy( m_evStop );
}

CGPUTestP2P::~CGPUTestP2P()
{
    if ( m_dptrDst ) {
        cudaSetDevice( m_dstDevice.getGPU() );
        cudaFree( m_dptrDst );
    }
    if ( m_dptrSrc ) {
        cudaSetDevice( m_srcDevice.getGPU() );
        cudaFree( m_dptrSrc );
    }
}

CGPUTestH2D::~CGPUTestH2D()
{
    if ( m_dptrDst ) {
        cudaSetDevice( m_dstDevice.getGPU() );
        cudaFree( m_dptrDst );
    }
    chNUMApageAlignedFreeHost( m_pSrc );
}

CGPUTestD2H::~CGPUTestD2H()
{
    if ( m_dptrSrc ) {
        cudaSetDevice( m_srcDevice.getGPU() );
        cudaFree( m_dptrSrc );
    }
    cudaFreeHost( m_pDst );
}

bool
CGPUTestP2P::PerformMemcpys( )
{
    bool bRet = false;
    cudaError_t status;
    for ( int j = 0; j < g_cIterations; j++ ) {
        cuda(MemcpyPeerAsync( m_dptrDst, m_dstDevice.getGPU(),
                                           m_dptrSrc, m_srcDevice.getGPU(),
                                           g_cBytes, NULL ) );
    }
    bRet = true;
Error:
    return bRet;
}

__global__ void
p2pPingPongLatencyTest( 
    void *_pLocal, 
    void *_pRemote, 
    uint64_t *pTimestamps,
    int bWait,
    int cIterations )
{
    volatile int *pLocal = (volatile int *) _pLocal;
    volatile int *pRemote = (volatile int *) _pRemote;
    int pingpongValue = 0;
    while ( cIterations-- ) {
        *pTimestamps++ = clock64();
        if ( bWait )
        while ( *pLocal != pingpongValue );
        bWait = 1;
        pingpongValue = 1-pingpongValue;
        *pRemote = pingpongValue;
    }
}

void
computeStatistics( int32_t *pmin, int32_t *pmax, double *pmean, double *pstdev, uint64_t *pClocks, size_t N )
{
    int32_t min = INT_MAX;
    int32_t max = INT_MIN;
    int64_t sum = 0, sumsq = 0;
    size_t cSamples = 0;
    for ( size_t i = 10; i < N-1; i++ ) {
        int32_t diff = (int32_t) (pClocks[i+1] - pClocks[i]);
        sum += diff;
        sumsq += (uint64_t) diff*diff;
        if ( diff < min ) min = diff;
        if ( diff > max ) max = diff;
        cSamples += 1;
    }
    *pmin = min;
    *pmax = max;
    *pmean = sum / (double) cSamples;
    int64_t numerator = (int64_t) cSamples*sumsq - sum*sum;
    *pstdev = sqrt(numerator / ((double) cSamples*(cSamples-1)));
}

bool
CGPUTestP2PLatency::PerformMemcpys( )
{
    bool bRet = false;
    cudaError_t status;
    cudaDeviceProp prop;
    double srcClockRate, dstClockRate;
    uint64_t *phostDst = (uint64_t *) malloc( (m_cIterations+1)*sizeof(uint64_t) );
    uint64_t *phostSrc = (uint64_t *) malloc( (m_cIterations+1)*sizeof(uint64_t) );
    cuda(SetDevice( m_srcDevice.getGPU() ) );
    cuda(GetDeviceProperties( &prop, m_srcDevice.getGPU() ) );
    srcClockRate = (double) prop.clockRate;
    p2pPingPongLatencyTest<<<1,1>>>( m_dptrSrc, m_dptrDst, m_dptrSrcTimestamps, 1, m_cIterations );
    cuda(SetDevice( m_dstDevice.getGPU() ) );
    cuda(GetDeviceProperties( &prop, m_dstDevice.getGPU() ) );
    dstClockRate = (double) prop.clockRate;
    p2pPingPongLatencyTest<<<1,1>>>( m_dptrDst, m_dptrSrc, m_dptrDstTimestamps, 0, m_cIterations );
    cuda(DeviceSynchronize() );
    cuda(Memcpy( phostDst, m_dptrDstTimestamps, (m_cIterations+1)*sizeof(uint64_t) , cudaMemcpyDeviceToHost ) );
    cuda(SetDevice( m_srcDevice.getGPU() ) );
    cuda(Memcpy( phostSrc, m_dptrSrc, sizeof(uint64_t) , cudaMemcpyDeviceToHost ) );
    cuda(Memcpy( phostSrc, m_dptrDstTimestamps, (m_cIterations+1)*sizeof(uint64_t), cudaMemcpyDeviceToHost ) );

    printf( "\nClocks statistics (dst):\n" );
    {
        int32_t minClocks, maxClocks;
        double meanClocks, stdevClocks;
        double clockRate = dstClockRate/1e6;
#if 0
        for ( int i = 0; i < m_cIterations-1; i++ ) {
            int clocks = phostDst[i+1]-phostDst[i];
            printf( "    %llX\t%d (%d Hz) = %.0f us\n", phostDst[i], clocks, srcClockRate, (double) clocks*1e6 / (double) dstClockRate );
        }
#endif
        computeStatistics( &minClocks, &maxClocks, &meanClocks, &stdevClocks, phostDst, m_cIterations );
        printf( "    min: %d clocks (%.0f ns)\n", minClocks, minClocks/clockRate );
        printf( "    max: %d clocks (%.0f ns)\n", maxClocks, maxClocks/clockRate );
        printf( "    mean: %.2f clocks (%.0f ns)\n", meanClocks, meanClocks/clockRate );
        printf( "    stdev: %.2f clocks (%.0f ns)\n", stdevClocks, stdevClocks/clockRate );
    }

    printf( "Clocks statistics (src):\n" );
    {
        int32_t minClocks, maxClocks;
        double meanClocks, stdevClocks;
        double clockRate = srcClockRate/1e6;
#if 0
        for ( int i = 0; i < m_cIterations-1; i++ ) {
            int clocks = phostSrc[i+1]-phostSrc[i];
            printf( "    %llX\t%d (%d Hz) = %.0f us\n", phostSrc[i], clocks, dstClockRate, (double) clocks*1e6 / (double) srcClockRate );
        }
#endif
        computeStatistics( &minClocks, &maxClocks, &meanClocks, &stdevClocks, phostSrc, m_cIterations );
        printf( "    min: %d clocks (%.0f ns)\n", minClocks, minClocks/clockRate );
        printf( "    max: %d clocks (%.0f ns)\n", maxClocks, maxClocks/clockRate );
        printf( "    mean: %.2f clocks (%.0f ns)\n", meanClocks, meanClocks/clockRate );
        printf( "    stdev: %.2f clocks (%.0f ns)\n", stdevClocks, stdevClocks/clockRate );
    }

    bRet = true;
Error:
    return bRet;
}

bool
CGPUTestH2D::PerformMemcpys( )
{
    bool bRet = false;
    cudaError_t status;
    cuda(SetDevice( m_dstDevice.getGPU() ) );
    for ( int j = 0; j < g_cIterations; j++ ) {
        cuda(MemcpyAsync( m_dptrDst, m_pSrc, g_cBytes, cudaMemcpyHostToDevice ) );
    }
    bRet = true;
Error:
    return bRet;
}

bool
CGPUTestD2H::PerformMemcpys( )
{
    bool bRet = false;
    cudaError_t status;
    cuda(SetDevice( m_srcDevice.getGPU() ) );
    for ( int j = 0; j < g_cIterations; j++ ) {
        cuda(MemcpyAsync( m_pDst, m_dptrSrc, g_cBytes, cudaMemcpyDeviceToHost ) );
    }
    bRet = true;
Error:
    return bRet;
}

bool
CGPULoadDriver::TimeMemcpys( )
{
    cudaError_t status;
    bool bRet = false;
    bool bAcquiredMutex = false;

    CHECK_NONZERO( sem_wait( &m_semWait ) );
    cuda(SetDevice( m_eventDevice.getGPU() ) );
    if ( m_bUseEvents ) {
        cuda(EventRecord( m_evStart, NULL ) );
    }
    if ( ! PerformMemcpys() )
        goto Error;
    if ( m_bUseEvents ) {
        cuda(SetDevice( m_eventDevice.getGPU() ) );
        cuda(EventRecord( m_evStop, NULL ) );
    }
    cuda(DeviceSynchronize() );

    if ( m_bUseEvents ) {
        cuda(SetDevice( m_eventDevice.getGPU() ) );
        cuda(EventRecord( m_evStop, NULL ) );
    }
    cuda(DeviceSynchronize() );

    CHECK_NONZERO( pthread_mutex_lock( &g_mutexOutput ) );
    bAcquiredMutex = true;

    if ( m_bUseEvents && (! m_bLatencyTest) )
    {
        float ms;
        cuda(EventElapsedTime( &ms, m_evStart, m_evStop ) );
        double MBytes = g_cIterations*g_cBytes / 1048576.0;
        double MBpers = 1000.0*MBytes / ms;

        cout << "    " << m_dstDevice << " <- " << m_srcDevice << ": " << MBpers << " MB/s (CUDA event)" << endl;
    }
    bRet = true;
Error:
    CHECK_NONZERO( sem_post( &m_semDone ) );
    if ( bAcquiredMutex ) {
        CHECK_NONZERO( pthread_mutex_unlock( &g_mutexOutput ) );
    }
    return bRet;
}

void *
CGPULoadDriver::ThreadProc( void *pContext )
{
    CGPULoadDriver *p = (CGPULoadDriver *) pContext;

    CHECK_NONZERO( sem_wait( &p->m_semWait ) );
    p->TimeMemcpys( );
    CHECK_NONZERO( sem_post( &p->m_semDone ) );
    return NULL;
}

bool
LaunchMemcpys_threaded( vector<GPUPair> pairs, size_t cBytes, bool bUseEvents )
{
    int cPairs = pairs.size();
    chTimerTimestamp start, stop;

    pthread_t *threads = new pthread_t[cPairs];
    vector< CGPULoadDriver * > tests( cPairs );

    for ( int i = 0; i < cPairs; i++ ) {
        // CPU to CPU transfers not supported
        if ( pairs[i].iSrc.bCPU() && pairs[i].iDst.bCPU() ) {
            return false;
        }
        tests[i] = makeLoadDriver( pairs[i].iDst, pairs[i].iSrc, cBytes, bUseEvents, pairs[i].m_bLatencyTest );
    }

    for ( int i = 0; i < cPairs; i++ ) {
        CHECK_NONZERO( pthread_create( &threads[i], NULL, CGPULoadDriver::ThreadProc, (void *) tests[i] ) );
    }
    sleep(1); // let the threads get a chance to hit the semaphore wait

    chTimerGetTime( &start );
    for ( int i = 0; i < cPairs; i++ ) {
        CHECK_NONZERO( sem_post( &tests[i]->m_semWait ) );
    }
    for ( int i = 0; i < cPairs; i++ ) {
        CHECK_NONZERO( sem_wait( &tests[i]->m_semDone ) );
    }
    for ( int i = 0; i < cPairs; i++ ) {
        pthread_join( threads[i], NULL );
    }
    chTimerGetTime( &stop );

    {
        int cActivePairs = 0;
        for ( int i = 0; i < cPairs; i++ ) {
            cActivePairs += ! pairs[i].m_bLatencyTest;
        }

        double TotalMBytes = (double) cActivePairs  * g_cBytes * g_cIterations / 1e6;
        if ( cActivePairs != 0 ) {
            double ElapsedTime = chTimerElapsedTime( &start, &stop );
            printf( "    Wall clock total observed bandwidth%s: %.0f MB/s\n", bUseEvents?"":" (no events)", TotalMBytes/ElapsedTime );
        }
    }
    for ( int i = 0; i < cPairs; i++ ) {
        delete tests[i];
    }
    return true;
}

bool
RunTest( vector<GPUPair> pairs, size_t cBytes, const char *szTestName )
{
    cout << szTestName << " test:" << endl;;
    for ( int i = 0; i < pairs.size(); i++ ) {
        cout << "    " << pairs[i] << endl;
    }
    if ( ! LaunchMemcpys_threaded( pairs, cBytes, true ) ) goto Error;
// the bool says whether to use events.
//    if ( ! LaunchMemcpys_threaded( pairs, cBytes, false ) ) goto Error;
    return true;
Error:
    return false;
}

vector<GPUPair>
ReadConfigFile( const char *s )
{
    vector<GPUPair> v;
    ifstream cfgfile(s);
    if ( ! cfgfile ) {
        fprintf( stderr, "Could not open %s\n", s );
        exit(1);
    }
    while ( cfgfile ) {
        bool bSrcCPU, bDstCPU;
        int srcDevice, dstDevice;
        string deviceType;
        bool bLatencyTest = false;

        cfgfile >> deviceType;
        if ( deviceType=="latency" ) {
            bLatencyTest = true;
            cfgfile >> deviceType;
        }

        cfgfile >> dstDevice;

        if ( cfgfile.eof() )
            break;

        bDstCPU = false;
        if ( deviceType=="cpu" ) {
            if ( bLatencyTest ) {
                fprintf( stderr, "Latency test must be between GPUs\n" );
                exit(1);
            }
            bDstCPU = true;
        }
        else if ( deviceType != "gpu") {
            fprintf( stderr, "Parsing error (not cpu or gpu)\n" );
            exit(1);
        }
        {
            string check;
            cfgfile >> check;
            if ( check != "<-" ) {
                fprintf( stderr, "Parsing error (missing <- symbol)\n" );
                exit(1);
            }
        }
        cfgfile >> deviceType >> srcDevice;
        bSrcCPU = false;
        if ( deviceType=="cpu" ) {
            if ( bLatencyTest ) {
                fprintf( stderr, "Latency test must be between GPUs\n" );
                exit(1);
            }
            bSrcCPU = true;
        }
        else if ( deviceType != "gpu") {
            fprintf( stderr, "Parsing error (not cpu or gpu)\n" );
            exit(1);
        }
        v.push_back( GPUPair( cEnumCPUGPU(dstDevice, bDstCPU), cEnumCPUGPU( srcDevice, bSrcCPU), bLatencyTest ) );
    }

    return v;
}

int
main( int argc, char *argv[] )
{
    int deviceCount;

    cudaError_t status;

    printf( "Peer-to-peer memcpy... " ); fflush( stdout );

    cuda(GetDeviceCount( &deviceCount ) );

    if ( deviceCount <= 1 ) {
        printf( "Peer-to-peer demo requires at least 2 devices\n" );
        exit(1);
    }

    printf( "%d devices detected\n", deviceCount );

    pthread_mutex_init( &g_mutexOutput, NULL );

    for ( int i = 0; i < deviceCount; i++ ) {
        cudaSetDevice( i );
        for ( int j = 0; j < deviceCount; j++ ) {
            if ( i != j ) {
                int bEnabled;
                cuda(DeviceCanAccessPeer( &bEnabled, i, j ) );
                g_bEnabled[i][j] = (0 != bEnabled);
                if ( bEnabled ) {
                    cuda(DeviceEnablePeerAccess( j, 0 ) );
                }
            }
        }
    }

    if ( 2 != argc ) {
        fprintf( stderr, "Usage: %s <configfile>\n", argv[0] );
        exit(1);
    }

    {
        vector<GPUPair> v = ReadConfigFile( argv[1] );
        RunTest( v, g_cBytes, argv[1] );
    }

    return 0;
Error:
    printf( "Error\n" );
    return 1;
}
