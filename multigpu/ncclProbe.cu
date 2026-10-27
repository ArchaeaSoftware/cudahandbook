/*
 *
 * ncclProbe.cu
 *
 * Reports what a multi-GPU machine can actually do, and proves NCCL works on it.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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
// Run this first on an unfamiliar multi-GPU machine.  It answers, in one
// command, the questions that otherwise get answered expensively by a
// benchmark that quietly measured the wrong thing:
//
//   - How many devices are visible, and what are they?
//   - Which pairs can reach each other directly, and over what link?
//   - Does a collective actually produce the right answer here?
//
// The peer matrix matters because a rented instance may or may not have the
// interconnect its name implies, and cudaMemcpyPeerAsync succeeds either way:
// without peer access the driver stages the transfer through host memory, at
// a fraction of the bandwidth, with no error to notice.  A run that reports
// "no direct peer access" has explained its own disappointing numbers.
//
// The all-gather is checked rather than merely executed.  Each rank fills its
// contribution with its own rank number, so a correct gather leaves every
// rank holding every rank's number in rank order, and a wrong one says which
// rank's block went missing.
//

#include <stdio.h>
#include <stdlib.h>

#include <chError.h>
#include <chErrorNccl.h>

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    ncclResult_t ncclStatus;
    int ret = 1;

    int cDevices = 0;
    int version = 0;
    int driverVersion = 0;
    int runtimeVersion = 0;

    ncclComm_t *comms = NULL;
    int *devices = NULL;
    float **dptrSend = NULL;
    float **dptrRecv = NULL;
    cudaStream_t *streams = NULL;
    float *host = NULL;
    int cCommsCreated = 0;

    const size_t cElements = 1024;

    nccl(GetVersion( &version ) );
    cuda(GetDeviceCount( &cDevices ) );
    cuda(DriverGetVersion( &driverVersion ) );
    cuda(RuntimeGetVersion( &runtimeVersion ) );

    printf( "NCCL %d.%d.%d, CUDA driver %d.%d, runtime %d.%d\n",
        version/10000, (version/100)%100, version%100,
        driverVersion/1000, (driverVersion%1000)/10,
        runtimeVersion/1000, (runtimeVersion%1000)/10 );

    if ( 0 == cDevices ) {
        fprintf( stderr, "no CUDA devices\n" );
        goto Error;
    }

    for ( int i = 0; i < cDevices; i++ ) {
        cudaDeviceProp prop;
        cuda(GetDeviceProperties( &prop, i ) );
        printf( "  device %d: %s, SM %d.%d, %zu MiB, %d SMs\n",
            i, prop.name, prop.major, prop.minor,
            prop.totalGlobalMem/(1024*1024), prop.multiProcessorCount );
    }

    //
    // The peer matrix.  A row is a source device, a column a destination:
    // "-" on the diagonal, "P2P" where the source can address the
    // destination's memory directly, and "host" where it cannot and the
    // driver would stage the copy through system memory.
    //
    printf( "\npeer access (row = from, column = to):\n     " );
    for ( int j = 0; j < cDevices; j++ ) {
        printf( "%6d", j );
    }
    printf( "\n" );
    for ( int i = 0; i < cDevices; i++ ) {
        printf( "%5d", i );
        for ( int j = 0; j < cDevices; j++ ) {
            int bCanAccess = 0;
            if ( i == j ) {
                printf( "%6s", "-" );
                continue;
            }
            cuda(DeviceCanAccessPeer( &bCanAccess, i, j ) );
            printf( "%6s", bCanAccess ? "P2P" : "host" );
        }
        printf( "\n" );
    }

    //
    // NVLink shows up as a nonzero performance rank between two devices.  A
    // machine whose peer matrix is all P2P may still be routing over PCIe,
    // which this distinguishes.
    //
    printf( "\nNVLink performance rank (0 = not NVLink):\n     " );
    for ( int j = 0; j < cDevices; j++ ) {
        printf( "%6d", j );
    }
    printf( "\n" );
    for ( int i = 0; i < cDevices; i++ ) {
        printf( "%5d", i );
        for ( int j = 0; j < cDevices; j++ ) {
            int perfRank = 0;
            if ( i == j ) {
                printf( "%6s", "-" );
                continue;
            }
            // A device pair that is not linked reports an error rather than a
            // zero, so ask without the checking macro.
            if ( cudaSuccess != cudaDeviceGetP2PAttribute( &perfRank,
                     cudaDevP2PAttrPerformanceRank, i, j ) ) {
                perfRank = 0;
            }
            printf( "%6d", perfRank );
        }
        printf( "\n" );
    }

    //
    // Now the collective itself, one rank per device.
    //
    comms = (ncclComm_t *) calloc( cDevices, sizeof(ncclComm_t) );
    devices = (int *) calloc( cDevices, sizeof(int) );
    dptrSend = (float **) calloc( cDevices, sizeof(float *) );
    dptrRecv = (float **) calloc( cDevices, sizeof(float *) );
    streams = (cudaStream_t *) calloc( cDevices, sizeof(cudaStream_t) );
    host = (float *) malloc( cDevices*cElements*sizeof(float) );
    if ( ! comms || ! devices || ! dptrSend || ! dptrRecv || ! streams || ! host ) {
        fprintf( stderr, "out of memory\n" );
        goto Error;
    }

    for ( int i = 0; i < cDevices; i++ ) {
        devices[i] = i;
        cuda(SetDevice( i ) );
        cuda(Malloc( &dptrSend[i], cElements*sizeof(float) ) );
        cuda(Malloc( &dptrRecv[i], cDevices*cElements*sizeof(float) ) );
        cuda(StreamCreate( &streams[i] ) );
        for ( size_t k = 0; k < cElements; k++ ) {
            host[k] = (float) i;
        }
        cuda(Memcpy( dptrSend[i], host, cElements*sizeof(float),
            cudaMemcpyHostToDevice ) );
    }

    nccl(CommInitAll( comms, cDevices, devices ) );
    cCommsCreated = cDevices;

    // One rank per device from a single thread has to be enqueued as a group,
    // or the first call blocks waiting for ranks that this thread has not
    // issued yet.
    nccl(GroupStart() );
    for ( int i = 0; i < cDevices; i++ ) {
        nccl(AllGather( dptrSend[i], dptrRecv[i], cElements, ncclFloat,
            comms[i], streams[i] ) );
    }
    nccl(GroupEnd() );

    for ( int i = 0; i < cDevices; i++ ) {
        cuda(SetDevice( i ) );
        cuda(StreamSynchronize( streams[i] ) );
    }

    {
        int cErrors = 0;
        for ( int i = 0; i < cDevices; i++ ) {
            cuda(SetDevice( i ) );
            cuda(Memcpy( host, dptrRecv[i], cDevices*cElements*sizeof(float),
                cudaMemcpyDeviceToHost ) );
            for ( int blk = 0; blk < cDevices; blk++ ) {
                for ( size_t k = 0; k < cElements; k++ ) {
                    if ( (float) blk != host[blk*cElements+k] ) {
                        fprintf( stderr,
                            "rank %d: block %d is wrong at element %zu "
                            "(expected %d, got %f)\n",
                            i, blk, k, blk, host[blk*cElements+k] );
                        cErrors += 1;
                        break;
                    }
                }
            }
        }
        printf( "\nall-gather across %d rank(s): %s\n",
            cDevices, cErrors ? "FAILED" : "correct" );
        if ( 0 == cErrors ) {
            ret = 0;
        }
    }

Error:
    for ( int i = 0; i < cCommsCreated; i++ ) {
        if ( comms[i] ) {
            ncclCommDestroy( comms[i] );
        }
    }
    for ( int i = 0; i < cDevices; i++ ) {
        cudaSetDevice( i );
        if ( dptrSend && dptrSend[i] ) {
            cudaFree( dptrSend[i] );
        }
        if ( dptrRecv && dptrRecv[i] ) {
            cudaFree( dptrRecv[i] );
        }
        if ( streams && streams[i] ) {
            cudaStreamDestroy( streams[i] );
        }
    }
    free( comms );
    free( devices );
    free( dptrSend );
    free( dptrRecv );
    free( streams );
    free( host );
    return ret;
}
