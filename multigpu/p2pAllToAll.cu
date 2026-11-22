/*
 *
 * p2pAllToAll.cu
 *
 * Measures the cost of a peer-to-peer transfer between every ordered pair of
 * devices, one pair at a time and then all pairs at once.
 *
 * Copyright (c) 2026, Archaea Software, LLC.
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

//
// NVIDIA's p2pBandwidthLatencyTest already reports what one pair of devices
// can do while the rest of the machine is idle.  The question that sample
// does not answer, and the one a collective actually depends on, is what
// happens when every pair transfers at once.
//
// A switched fabric and a PCIe tree are indistinguishable one pair at a time:
// both report a respectable number, because a single transfer has the whole
// machine to itself.  They diverge under load.  An NVSwitch machine provides
// full bisection bandwidth, so N(N-1) simultaneous transfers each proceed at
// close to the rate a lone transfer saw.  A PCIe tree funnels every pair that
// does not share a switch through a common root, so the same traffic pattern
// collapses to a fraction of it.  That difference is why the ring
// decomposition in Chapter 9 can overlap its hops on one machine and
// serializes on another, and it is invisible to a pairwise measurement.
//
// So this program measures both and prints the ratio:
//
//   - idle: each ordered pair in turn, nothing else running.
//   - concurrent: every ordered pair at once, each on its own stream.
//
// Both numbers are the marginal cost of one transfer when the copy engines
// are kept fed -- iterations are issued back to back on a stream, so the cost
// reported is per-transfer throughput rather than the round-trip latency of a
// single isolated copy.  A small --bytes puts that cost in the regime where
// the fixed overhead of a transfer dominates, which is the latency question;
// a large --bytes puts it in the regime where the wire does, which is the
// bandwidth question.  The contention the program is built to expose shows up
// in the second regime, so it is worth running at both.
//
// The transfers are verified before any of them are timed.  Each device fills
// its send buffer with its own device number, so after a correct all-to-all
// every device holds every other device's number in the slot belonging to
// that device.  A transfer that silently went nowhere leaves a zero, and the
// program says which pair produced it rather than reporting a fast wrong
// answer.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <chrono>

#include <chError.h>
#include <chCommandLine.h>

const int g_maxDevices = 16;

//
// Writes one value into every element, so that a receiving device can say
// which device a buffer came from.
//
__global__ void
FillPattern( uint32_t *out, size_t N, uint32_t value )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x )
    {
        out[i] = value;
    }
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    int cDevices = 0;
    int iterations = 200;
    int bytes = 4096;
    size_t cb = 0;
    size_t cElements = 0;
    int cPairs = 0;
    int cStaged = 0;
    int cMismatch = 0;
    double idleTotal = 0.0;
    double idleMean = 0.0;
    double aggregate = 0.0;
    double onePair = 0.0;
    double elapsed = 0.0;

    uint32_t *dptrSend[g_maxDevices] = {0};
    uint32_t *dptrRecv[g_maxDevices] = {0};
    uint32_t *hostRecv = NULL;
    cudaStream_t streams[g_maxDevices][g_maxDevices] = {};
    bool bPeer[g_maxDevices][g_maxDevices] = {};
    float idle[g_maxDevices][g_maxDevices] = {};

    std::chrono::steady_clock::time_point start, stop;

    cuda(GetDeviceCount( &cDevices ) );
    chCommandLineGet( &cDevices, "numgpus", argc, argv );
    chCommandLineGet( &iterations, "iterations", argc, argv );
    chCommandLineGet( &bytes, "bytes", argc, argv );

    if ( cDevices > g_maxDevices ) {
        cDevices = g_maxDevices;
    }
    if ( 2 > cDevices ) {
        fprintf( stderr, "p2pAllToAll: needs at least 2 devices, found %d\n",
            cDevices );
        return 1;
    }
    if ( 0 != bytes % (int) sizeof(uint32_t) ) {
        fprintf( stderr, "p2pAllToAll: --bytes must be a multiple of %zu\n",
            sizeof(uint32_t) );
        return 1;
    }
    cb = (size_t) bytes;
    cElements = cb / sizeof(uint32_t);
    cPairs = cDevices * (cDevices - 1);

    printf( "p2pAllToAll: %d devices, %zu bytes per transfer, "
        "%d iterations, %d ordered pairs\n",
        cDevices, cb, iterations, cPairs );

    //
    // Open peer access wherever the hardware offers it.  Where it does not,
    // cudaMemcpyPeerAsync still succeeds by staging through host memory, so
    // the count of staged pairs is printed: a machine with no peer access at
    // all produces numbers that describe its host bus, not its fabric.
    //
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        for ( int j = 0; j < cDevices; j++ ) {
            int bCanAccess = 0;
            if ( i == j ) {
                continue;
            }
            cuda(DeviceCanAccessPeer( &bCanAccess, i, j ) );
            bPeer[i][j] = (0 != bCanAccess);
            if ( bPeer[i][j] ) {
                status = cudaDeviceEnablePeerAccess( j, 0 );
                if ( (cudaSuccess != status) &&
                     (cudaErrorPeerAccessAlreadyEnabled != status) ) {
                    fprintf( stderr,
                        "p2pAllToAll: enabling peer access %d->%d failed\n",
                        i, j );
                    goto Error;
                }
            }
            else {
                cStaged += 1;
            }
        }
    }
    printf( "peer access: %d of %d ordered pairs direct, %d staged via host\n",
        cPairs - cStaged, cPairs, cStaged );

    //
    // One send buffer per device, and one receive slot per device per peer,
    // so that every transfer in the all-to-all has a destination of its own
    // and nothing is overwritten by a concurrent arrival.
    //
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        cuda(Malloc( (void **) &dptrSend[i], cb ) );
        cuda(Malloc( (void **) &dptrRecv[i], cb * cDevices ) );
        cuda(Memset( dptrRecv[i], 0, cb * cDevices ) );
        FillPattern<<<256, 256>>>( dptrSend[i], cElements, (uint32_t) (i + 1) );
        cuda(GetLastError() );
        for ( int j = 0; j < cDevices; j++ ) {
            if ( i == j ) {
                continue;
            }
            cuda(StreamCreate( &streams[i][j] ) );
        }
    }
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        cuda(DeviceSynchronize() );
    }

    //
    // Correctness first, before any timing.
    //
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        for ( int j = 0; j < cDevices; j++ ) {
            if ( i == j ) {
                continue;
            }
            cuda(MemcpyPeerAsync( dptrRecv[j] + i*cElements, j,
                dptrSend[i], i, cb, streams[i][j] ) );
        }
    }
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        cuda(DeviceSynchronize() );
    }

    hostRecv = (uint32_t *) malloc( cb * cDevices );
    if ( ! hostRecv ) {
        fprintf( stderr, "p2pAllToAll: out of host memory\n" );
        goto Error;
    }
    for ( int j = 0; j < cDevices; j++ ) {
        cuda(SetDevice( j ) );
        cuda(Memcpy( hostRecv, dptrRecv[j], cb * cDevices,
            cudaMemcpyDeviceToHost ) );
        for ( int i = 0; i < cDevices; i++ ) {
            uint32_t expected = (uint32_t) (i + 1);
            if ( i == j ) {
                continue;
            }
            for ( size_t k = 0; k < cElements; k++ ) {
                if ( expected != hostRecv[i*cElements + k] ) {
                    fprintf( stderr,
                        "VERIFY FAILURE: device %d slot %d element %zu "
                        "holds %u, expected %u\n",
                        j, i, k, hostRecv[i*cElements + k], expected );
                    cMismatch += 1;
                    break;
                }
            }
        }
    }
    if ( 0 != cMismatch ) {
        fprintf( stderr, "p2pAllToAll: %d pair(s) did not deliver\n",
            cMismatch );
        goto Error;
    }
    printf( "verify: all %d ordered pairs delivered\n", cPairs );

    //
    // Idle: one ordered pair at a time, timed on the source device's stream.
    //
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        for ( int j = 0; j < cDevices; j++ ) {
            cudaEvent_t evStart, evStop;
            float ms = 0.0f;
            if ( i == j ) {
                continue;
            }
            cuda(EventCreate( &evStart ) );
            cuda(EventCreate( &evStop ) );
            // Warm up, so the first transfer's one-time costs land outside
            // the timed region.
            for ( int k = 0; k < 8; k++ ) {
                cuda(MemcpyPeerAsync( dptrRecv[j] + i*cElements, j,
                    dptrSend[i], i, cb, streams[i][j] ) );
            }
            cuda(StreamSynchronize( streams[i][j] ) );
            cuda(EventRecord( evStart, streams[i][j] ) );
            for ( int k = 0; k < iterations; k++ ) {
                cuda(MemcpyPeerAsync( dptrRecv[j] + i*cElements, j,
                    dptrSend[i], i, cb, streams[i][j] ) );
            }
            cuda(EventRecord( evStop, streams[i][j] ) );
            cuda(EventSynchronize( evStop ) );
            cuda(EventElapsedTime( &ms, evStart, evStop ) );
            idle[i][j] = ms * 1000.0f / (float) iterations;
            idleTotal += idle[i][j];
            cuda(EventDestroy( evStart ) );
            cuda(EventDestroy( evStop ) );
        }
    }
    idleMean = idleTotal / (double) cPairs;

    printf( "\nidle: microseconds per transfer, one pair at a time\n" );
    printf( "        " );
    for ( int j = 0; j < cDevices; j++ ) {
        printf( "%8d", j );
    }
    printf( "\n" );
    for ( int i = 0; i < cDevices; i++ ) {
        printf( "%7d", i );
        for ( int j = 0; j < cDevices; j++ ) {
            if ( i == j ) {
                printf( "%8s", "-" );
                continue;
            }
            printf( "%8.2f", idle[i][j] );
        }
        printf( "\n" );
    }

    //
    // Concurrent: every ordered pair at once.  The whole pattern is timed on
    // the host, because what matters is when the last transfer of a round
    // finishes on any device, which no single device's events can observe.
    //
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        cuda(DeviceSynchronize() );
    }
    start = std::chrono::steady_clock::now();
    for ( int k = 0; k < iterations; k++ ) {
        for ( int i = 0; i < cDevices; i++ ) {
            cuda(SetDevice( i ) );
            for ( int j = 0; j < cDevices; j++ ) {
                if ( i == j ) {
                    continue;
                }
                cuda(MemcpyPeerAsync( dptrRecv[j] + i*cElements, j,
                    dptrSend[i], i, cb, streams[i][j] ) );
            }
        }
    }
    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        cuda(DeviceSynchronize() );
    }
    stop = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration<double>( stop - start ).count();

    aggregate = (double) cb * (double) cPairs * (double) iterations /
        elapsed / 1.0e9;
    onePair = (double) cb / (idleMean * 1.0e-6) / 1.0e9;

    printf( "\nconcurrent: all %d ordered pairs at once\n", cPairs );
    printf( "  one pair alone:    %8.2f GB/s  (%6.2f us per transfer)\n",
        onePair, idleMean );
    printf( "  all pairs at once: %8.2f GB/s aggregate, "
        "%7.2f GB/s per device out\n",
        aggregate,
        aggregate / (double) cDevices );
    printf( "  concurrency gain:  %8.2fx a single pair "
        "(%.0fx would be every device sending at its solo rate)\n",
        aggregate / onePair,
        (double) cDevices );

    //
    // The gain is the number that separates a crossbar from a tree, and it is
    // deliberately not a comparison of per-transfer times.  Divide the
    // concurrent wall clock by the number of transfers and the result comes
    // out *below* the idle time per transfer, because the transfers overlap;
    // read as a speedup it says nothing, and read as a slowdown it says
    // something false.  Aggregate bandwidth against one pair's bandwidth is
    // the honest comparison: a gain near the device count means every device
    // sent at nearly its solo rate while every other device did too, and a
    // gain near 1 means the pairs were all queueing for the same wire.
    //
    // Expect neither extreme.  A device sending to every peer at once is also
    // receiving from every peer at once, so its ports carry twice the traffic
    // the outbound figure names, and the copy engines serialize some of what
    // the fabric could have carried in parallel.
    //
    status = cudaSuccess;

Error:
    free( hostRecv );
    for ( int i = 0; i < cDevices; i++ ) {
        if ( cudaSuccess == cudaSetDevice( i ) ) {
            for ( int j = 0; j < cDevices; j++ ) {
                if ( streams[i][j] ) {
                    cudaStreamDestroy( streams[i][j] );
                }
            }
            cudaFree( dptrSend[i] );
            cudaFree( dptrRecv[i] );
        }
    }
    return (cudaSuccess == status) ? 0 : 1;
}
