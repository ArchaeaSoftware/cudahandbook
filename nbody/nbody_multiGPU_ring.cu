/*
 *
 * nbody_multiGPU_ring.cu
 *
 * Multi-GPU N-body calculation that exchanges positions in a ring.
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
// Every device needs every body's position, but the host-staged
// implementation in nbody_multiGPU.cu gets them the most expensive way
// available: each device pulls the whole 4*N-float array across its own PCIe
// link, every timestep.  The host bus carries g_numGPUs copies of the data
// and the devices sit idle while it does.
//
// This implementation moves the same information as a ring all-gather, which
// is how NCCL implements the collective.  Each device uploads only the block
// of bodies it owns -- 4*N/g_numGPUs floats, so the host bus carries the
// array once rather than g_numGPUs times -- and then the devices pass blocks
// around the ring among themselves.  Device i sends to device i+1 and
// receives from device i-1, so after g_numGPUs-1 hops every device holds
// every block.
//
// The total bytes moved does not fall: g_numGPUs-1 hops of N/g_numGPUs
// bodies apiece is still O(N) per device.  Two other things change, and they
// are the point.  The traffic leaves the host bus for the peer interconnect,
// which on an NVLink or NVSwitch machine is an order of magnitude faster.
// And because a device can start computing against a block the moment that
// block arrives, the hops overlap with the kernels instead of preceding
// them.
//

#include <stdio.h>
#include <string.h>

#include <chError.h>
#include <chrono>

#include "nbody.h"
#include "nbody_multiGPU_buffers.h"

#include "bodybodyInteraction.cuh"

// The ring launches one kernel per arriving block instead of one kernel per
// timestep, so the kernel takes the source block as a parameter and
// accumulates into the force array rather than overwriting it.
__global__ void
ComputeNBodyGravitation_Ring( 
    float *force, 
    float *posMass, 
    float softeningSquared, 
    size_t base,        // index of this device's first body
    size_t n,           // number of bodies this device owns
    size_t srcBase,     // index of the source block's first body
    size_t srcN )       // number of bodies in the source block
{
    extern __shared__ float4 shPosMass[];
    for ( int m = blockIdx.x*blockDim.x + threadIdx.x;
              m < n;
              m += blockDim.x*gridDim.x )
    {
        size_t i = base+m;
        float acc[3] = {0};
        float4 myPosMass = ((float4 *) posMass)[i];
        for ( size_t j = 0; j < srcN; j += blockDim.x ) {
            shPosMass[threadIdx.x] = ((float4 *) posMass)[srcBase+j+threadIdx.x];
            __syncthreads();
            for ( size_t k = 0; k < blockDim.x; k++ ) {
                float fx, fy, fz;
                float4 bodyPosMass = shPosMass[k];

                bodyBodyInteraction( 
                    &fx, &fy, &fz, 
                    myPosMass.x, myPosMass.y, myPosMass.z, 
                    bodyPosMass.x, 
                    bodyPosMass.y, 
                    bodyPosMass.z, 
                    bodyPosMass.w, 
                    softeningSquared );
                acc[0] += fx;
                acc[1] += fy;
                acc[2] += fz;
            }
            __syncthreads();
        }
        force[3*m+0] += acc[0];
        force[3*m+1] += acc[1];
        force[3*m+2] += acc[2];
    }
}

// Each device gets two streams, because a hop has to overlap with the kernels
// rather than take turns with them: one stream carries the transfers and one
// carries the kernels, and events order the two against each other.  A single
// stream per device would serialize the copy for hop h against the kernel for
// hop h, which is the overlap this decomposition exists to get.
//
// hopDone[i][h] fires when device i has finished sending its hop-h block, so
// device i+1 can wait on it before reading or forwarding that block.
// uploadDone[i] fires when device i's own block has arrived from the host.
struct ringContext {
    cudaStream_t copyStream[g_maxGPUs];
    cudaStream_t computeStream[g_maxGPUs];
    cudaEvent_t  hopDone[g_maxGPUs][g_maxGPUs];
    cudaEvent_t  uploadDone[g_maxGPUs];
    int          numGPUs;
    int          cDirectLinks;      // ring hops that peer access made direct
};

static ringContext g_ring;
static const int g_ringBlockSize = 256;
static const int g_ringMaxBlocks = 300;
static bool g_bRingFirstStep = true;

void
multiGPURingRelease( void )
{
    int oldDevice;
    if ( cudaSuccess != cudaGetDevice( &oldDevice ) ) {
        return;
    }
    for ( int i = 0; i < g_ring.numGPUs; i++ ) {
        cudaSetDevice( i );
        for ( int h = 0; h < g_ring.numGPUs; h++ ) {
            if ( g_ring.hopDone[i][h] ) {
                cudaEventDestroy( g_ring.hopDone[i][h] );
                g_ring.hopDone[i][h] = NULL;
            }
        }
        if ( g_ring.uploadDone[i] ) {
            cudaEventDestroy( g_ring.uploadDone[i] );
            g_ring.uploadDone[i] = NULL;
        }
        if ( g_ring.copyStream[i] ) {
            cudaStreamDestroy( g_ring.copyStream[i] );
            g_ring.copyStream[i] = NULL;
        }
        if ( g_ring.computeStream[i] ) {
            cudaStreamDestroy( g_ring.computeStream[i] );
            g_ring.computeStream[i] = NULL;
        }
    }
    cudaSetDevice( oldDevice );
    g_ring.numGPUs = 0;
    g_ring.cDirectLinks = 0;
}

// Creates the streams and events, and enables peer access along the ring.
// Peer access is what makes a hop a direct device-to-device transfer; without
// it cudaMemcpyPeerAsync still works, but the driver stages the data through
// host memory, which is the cost this decomposition exists to avoid.  The
// count of direct links is reported so that a run on a machine without a peer
// interconnect is not mistaken for one with it.
static bool
multiGPURingAcquire( int numGPUs )
{
    cudaError_t status;
    int oldDevice;
    bool bRet = false;

    if ( numGPUs == g_ring.numGPUs ) {
        return true;
    }
    multiGPURingRelease();
    g_bRingFirstStep = true;
    if ( numGPUs > g_maxGPUs ) {
        return false;
    }

    cuda(GetDevice( &oldDevice ) );
    for ( int i = 0; i < numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        // Count the device in before creating anything for it, so a failure
        // partway through tears down what this loop already created.
        g_ring.numGPUs = i+1;
        cuda(StreamCreate( &g_ring.copyStream[i] ) );
        cuda(StreamCreate( &g_ring.computeStream[i] ) );
        // The events only order work; they are never used for timing, so
        // disabling timing keeps them cheap.
        cuda(EventCreateWithFlags( &g_ring.uploadDone[i], cudaEventDisableTiming ) );
        for ( int h = 0; h < numGPUs; h++ ) {
            cuda(EventCreateWithFlags( &g_ring.hopDone[i][h], cudaEventDisableTiming ) );
        }
    }

    // Device i sends to its successor around the ring, so device i is the one
    // that needs access to the successor's memory.
    for ( int i = 0; i < numGPUs; i++ ) {
        int iNext = (i+1) % numGPUs;
        int bCanAccess = 0;
        if ( iNext == i ) {
            continue;
        }
        cuda(SetDevice( i ) );
        cuda(DeviceCanAccessPeer( &bCanAccess, i, iNext ) );
        if ( ! bCanAccess ) {
            continue;
        }
        status = cudaDeviceEnablePeerAccess( iNext, 0 );
        if ( (cudaSuccess == status) || 
             (cudaErrorPeerAccessAlreadyEnabled == status) ) {
            g_ring.cDirectLinks += 1;
        }
    }
    fprintf( stderr, "ring: %d of %d hops are direct peer transfers\n", 
        g_ring.cDirectLinks, 
        (1 == numGPUs) ? 0 : numGPUs );
    bRet = true;

Error:
    cudaSetDevice( oldDevice );
    if ( ! bRet ) {
        multiGPURingRelease();
    }
    return bRet;
}

// Confirms that the all-gather actually delivered every block to every
// device, by reading each device's position array back and comparing it with
// the host's copy.  A ring bug otherwise surfaces only as a wrong force,
// which is a long way from the hop that dropped the block; this reports the
// device and the block instead.  It runs on the first timestep of a run and
// on every timestep if --ringcheck is given, which is the difference between
// catching a wiring mistake and catching a race.
static bool
multiGPURingVerify( size_t N, size_t bodiesPerGPU )
{
    cudaError_t status;
    bool bRet = false;
    int oldDevice;
    float *check = NULL;

    cuda(GetDevice( &oldDevice ) );
    check = new float[4*N];
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        cuda(Memcpy( check, g_multiGPUBuffers[i].posMass, 4*N*sizeof(float), 
            cudaMemcpyDeviceToHost ) );
        for ( int blk = 0; blk < g_numGPUs; blk++ ) {
            size_t off = 4*bodiesPerGPU*blk;
            if ( memcmp( check+off, g_hostAOS_PosMass+off, 
                         4*bodiesPerGPU*sizeof(float) ) ) {
                fprintf( stderr, 
                    "ring: device %d never received block %d intact "
                    "(block %d of %d, %zu bodies at offset %zu)\n",
                    i, blk, blk, g_numGPUs, bodiesPerGPU, off/4 );
                goto Error;
            }
        }
    }
    bRet = true;

Error:
    delete[] check;
    cudaSetDevice( oldDevice );
    return bRet;
}

float
ComputeGravitation_multiGPU_ring( 
    float *force, 
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;

    float ret = 0.0f;
    int oldDevice;
    int cBlocks;

    std::chrono::steady_clock::time_point start, end;

    size_t bodiesPerGPU = N / g_numGPUs;
    if ( (0 != N % g_numGPUs) || (g_numGPUs > g_maxGPUs) ) {
        return 0.0f;
    }
    // The kernel's shared-memory staging loop reads a whole block of threads'
    // worth of bodies at a time, so a block of bodies has to be a multiple of
    // the thread count.
    if ( 0 != bodiesPerGPU % g_ringBlockSize ) {
        fprintf( stderr, 
            "ring: %zu bodies on %d GPUs gives %zu bodies per GPU, "
            "which is not a multiple of %d\n",
            N, g_numGPUs, bodiesPerGPU, g_ringBlockSize );
        return 0.0f;
    }
    if ( ! multiGPUBuffersAcquire( g_numGPUs, N ) ) {
        return 0.0f;
    }
    if ( ! multiGPURingAcquire( g_numGPUs ) ) {
        return 0.0f;
    }
    cuda(GetDevice( &oldDevice ) );

    // Launch only as many blocks as there is work for.  Each device owns
    // 1/g_numGPUs of the bodies, so a fixed grid sized for the whole problem
    // would leave most blocks with nothing to do.
    cBlocks = (int) ((bodiesPerGPU + g_ringBlockSize - 1) / g_ringBlockSize);
    if ( cBlocks > g_ringMaxBlocks ) {
        cBlocks = g_ringMaxBlocks;
    }

    start = std::chrono::steady_clock::now();

    // Upload each device's own block, and clear the force accumulator it is
    // about to add into.  The upload rides the copy stream and signals
    // uploadDone, because both the first kernel and the first hop read the
    // block it delivers.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        cuda(MemcpyAsync( 
            g_multiGPUBuffers[i].posMass + 4*bodiesPerGPU*i, 
            g_hostAOS_PosMass + 4*bodiesPerGPU*i, 
            4*bodiesPerGPU*sizeof(float), 
            cudaMemcpyHostToDevice, 
            g_ring.copyStream[i] ) );
        cuda(EventRecord( g_ring.uploadDone[i], g_ring.copyStream[i] ) );
        cuda(StreamWaitEvent( g_ring.computeStream[i], g_ring.uploadDone[i], 0 ) );
        cuda(MemsetAsync( 
            g_multiGPUBuffers[i].force, 
            0, 
            3*bodiesPerGPU*sizeof(float), 
            g_ring.computeStream[i] ) );
    }

    // At hop h, device i works on the block it received at hop h-1 and
    // forwards that same block to its successor.  Both the forward and the
    // kernel wait on the predecessor's hop h-1 event and then proceed
    // concurrently, so the transfer for one hop overlaps the arithmetic for
    // it.  Hop 0 is each device's own block, which needs no hop to arrive.
    for ( int h = 0; h < g_numGPUs; h++ ) {
        for ( int i = 0; i < g_numGPUs; i++ ) {
            int iPrev = (i + g_numGPUs - 1) % g_numGPUs;
            int iNext = (i + 1) % g_numGPUs;
            size_t srcBlock = (i + g_numGPUs - h) % g_numGPUs;

            cuda(SetDevice( i ) );
            if ( 0 != h ) {
                cuda(StreamWaitEvent( g_ring.copyStream[i], 
                                      g_ring.hopDone[iPrev][h-1], 0 ) );
                cuda(StreamWaitEvent( g_ring.computeStream[i], 
                                      g_ring.hopDone[iPrev][h-1], 0 ) );
            }
            // Pass the block along, unless this is the last hop, by which
            // point every device already holds every block.
            if ( h < g_numGPUs-1 ) {
                cuda(MemcpyPeerAsync( 
                    g_multiGPUBuffers[iNext].posMass + 4*bodiesPerGPU*srcBlock, 
                    iNext, 
                    g_multiGPUBuffers[i].posMass + 4*bodiesPerGPU*srcBlock, 
                    i, 
                    4*bodiesPerGPU*sizeof(float), 
                    g_ring.copyStream[i] ) );
                cuda(EventRecord( g_ring.hopDone[i][h], g_ring.copyStream[i] ) );
            }
            ComputeNBodyGravitation_Ring
                <<<cBlocks, g_ringBlockSize, g_ringBlockSize*sizeof(float4), 
                   g_ring.computeStream[i]>>>( 
                g_multiGPUBuffers[i].force,
                g_multiGPUBuffers[i].posMass,
                softeningSquared,
                bodiesPerGPU*i,
                bodiesPerGPU,
                bodiesPerGPU*srcBlock,
                bodiesPerGPU );
        }
    }

    // Bring each device's slice of the force array back for the host's
    // integration step.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        cuda(MemcpyAsync( 
            g_hostAOS_Force + 3*bodiesPerGPU*i, 
            g_multiGPUBuffers[i].force, 
            3*bodiesPerGPU*sizeof(float), 
            cudaMemcpyDeviceToHost, 
            g_ring.computeStream[i] ) );
    }
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        cuda(StreamSynchronize( g_ring.copyStream[i] ) );
        cuda(StreamSynchronize( g_ring.computeStream[i] ) );
    }
    end = std::chrono::steady_clock::now();
    ret = std::chrono::duration<double>(end - start).count() * 1000.0f;

    // Verify off the clock, so the check never inflates the timing.  Clear
    // the latch only on success: a check that disarms itself when it fails
    // reports the first bad timestep and then lets every later one through
    // looking healthy.
    if ( g_bRingFirstStep || g_bRingCheck ) {
        if ( ! multiGPURingVerify( N, bodiesPerGPU ) ) {
            return 0.0f;
        }
        g_bRingFirstStep = false;
    }

Error:
    cudaSetDevice( oldDevice );
    return ret;
}
