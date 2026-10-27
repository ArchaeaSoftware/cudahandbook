/*
 *
 * nbody_multiGPU_nccl.cu
 *
 * Two-dimensional force decomposition of the N-body calculation, over NCCL.
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
// The decompositions in nbody_multiGPU.cu and nbody_multiGPU_ring.cu both
// divide the bodies among the devices and give every device every position.
// That is what makes their communication grow as O(N*G): each of G devices
// must be told about all N bodies, however the telling is arranged.
//
// Arranging the devices as a two-dimensional grid breaks that.  With the
// devices in a Pr x Pc grid, device (r,c) computes one tile of the force
// matrix: the forces that source block c exerts on target block r.  It needs
// the positions of two blocks, N/Pr targets and N/Pc sources, rather than all
// N.  Summed over the grid and with Pr and Pc near sqrt(G), the positions
// moved fall from O(N*G) to O(N*sqrt(G)).
//
// The arithmetic does not change.  Device (r,c) evaluates (N/Pr)*(N/Pc) =
// N^2/G interactions, exactly what a device computed under the
// one-dimensional decomposition.  Only the communication falls, which is
// precisely the point: on enough devices the one-dimensional decomposition
// becomes a communication benchmark wearing an N-body costume.
//
// What the grid costs is a reduction.  Each device now holds only a partial
// force for its target block -- the contribution of its own source block --
// so the Pc devices along a row must sum their partials before the forces are
// complete.  That sum is an all-reduce, and it is the step that specialized
// interconnect hardware accelerates: NVSwitch carries accumulators that can
// perform the summation inside the switch rather than moving every partial to
// every device, reached through NCCL's NVLS algorithm.  The call is isolated
// in reduceRowForces below so that comparing algorithms is a matter of
// setting NCCL_ALGO and running the program again.
//

#include <stdio.h>

#include <chError.h>
#include <chErrorNccl.h>
#include <chrono>

#include "nbody.h"
#include "nbody_multiGPU_buffers.h"

#include "nbody_multiGPU_tile.cuh"

__global__ void
ComputeNBodyGravitation_2D( 
    float *force, 
    float *posMass, 
    float softeningSquared, 
    size_t base, 
    size_t n, 
    size_t srcBase, 
    size_t srcN )
{
    ComputeNBodyGravitation_Tile( 
        force, 
        posMass, 
        softeningSquared, 
        base, 
        n, 
        srcBase, 
        srcN );
}

struct ncclContext {
    ncclComm_t   world[g_maxGPUs];      // one rank per device
    ncclComm_t   row[g_maxGPUs];        // the devices sharing a target block
    cudaStream_t stream[g_maxGPUs];
    int          numGPUs;
    int          rows;
    int          cols;
};

static ncclContext g_nccl;
static const int g_2DBlockSize = 256;
static const int g_2DMaxBlocks = 300;

// The most nearly square factorization of the device count.  Computing it
// rather than fixing it means a run on six devices uses a 2x3 grid instead of
// refusing to start, and that a single device is simply a 1x1 grid, which is
// the case that can be tested without multi-GPU hardware.
static void
multiGPU2DGeometry( int numGPUs, int *pRows, int *pCols )
{
    int rows = 1;
    for ( int r = 1; r*r <= numGPUs; r++ ) {
        if ( 0 == numGPUs % r ) {
            rows = r;
        }
    }
    *pRows = rows;
    *pCols = numGPUs / rows;
}

void
multiGPU2DRelease( void )
{
    int oldDevice;
    if ( cudaSuccess != cudaGetDevice( &oldDevice ) ) {
        return;
    }
    for ( int i = 0; i < g_nccl.numGPUs; i++ ) {
        if ( g_nccl.row[i] ) {
            ncclCommDestroy( g_nccl.row[i] );
            g_nccl.row[i] = NULL;
        }
        if ( g_nccl.world[i] ) {
            ncclCommDestroy( g_nccl.world[i] );
            g_nccl.world[i] = NULL;
        }
        cudaSetDevice( i );
        if ( g_nccl.stream[i] ) {
            cudaStreamDestroy( g_nccl.stream[i] );
            g_nccl.stream[i] = NULL;
        }
    }
    cudaSetDevice( oldDevice );
    g_nccl.numGPUs = 0;
}

static bool
multiGPU2DAcquire( int numGPUs )
{
    cudaError_t status;
    ncclResult_t ncclStatus;
    int oldDevice;
    bool bRet = false;
    int devices[g_maxGPUs];

    if ( numGPUs == g_nccl.numGPUs ) {
        return true;
    }
    multiGPU2DRelease();
    if ( numGPUs > g_maxGPUs ) {
        return false;
    }
    multiGPU2DGeometry( numGPUs, &g_nccl.rows, &g_nccl.cols );

    cuda(GetDevice( &oldDevice ) );
    for ( int i = 0; i < numGPUs; i++ ) {
        devices[i] = i;
        cuda(SetDevice( i ) );
        g_nccl.numGPUs = i+1;
        cuda(StreamCreate( &g_nccl.stream[i] ) );
    }

    nccl(CommInitAll( g_nccl.world, numGPUs, devices ) );

    // Split the world communicator by row, so that each device's row
    // communicator contains exactly the devices that share its target block.
    // Driving several devices from one thread means the split has to be
    // issued for all of them as a group.
    nccl(GroupStart() );
    for ( int i = 0; i < numGPUs; i++ ) {
        int row = i / g_nccl.cols;
        cuda(SetDevice( i ) );
        nccl(CommSplit( g_nccl.world[i], row, i, &g_nccl.row[i], NULL ) );
    }
    nccl(GroupEnd() );

    fprintf( stderr, "\n2D: %d device(s) as a %dx%d grid\n", 
        numGPUs, g_nccl.rows, g_nccl.cols );
    bRet = true;

Error:
    cudaSetDevice( oldDevice );
    if ( ! bRet ) {
        multiGPU2DRelease();
    }
    return bRet;
}

//
// Sums the partial forces held by the devices along each row, leaving every
// device in a row with the complete force on that row's target block.
//
// This is the collective that NVSwitch's accumulators accelerate.  NCCL
// chooses how to perform it, and NCCL_ALGO overrides that choice, so the
// comparison between a ring reduction and a switch-resident one is made by
// running the program twice rather than by writing it twice.  An all-reduce
// leaves the answer on every device in the row where a reduce-scatter would
// leave each device a slice of it; the all-reduce is used here because it is
// the collective the hardware special-cases.
//
static ncclResult_t
reduceRowForces( size_t cFloatsPerRow )
{
    cudaError_t status;
    ncclResult_t ncclStatus = ncclSuccess;
    bool bGroupOpen = false;

    nccl(GroupStart() );
    bGroupOpen = true;
    for ( int i = 0; i < g_nccl.numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        // In place: the partial forces are summed over the row and the
        // result replaces them.
        nccl(AllReduce( 
            g_multiGPUBuffers[i].force, 
            g_multiGPUBuffers[i].force, 
            cFloatsPerRow, 
            ncclFloat, 
            ncclSum, 
            g_nccl.row[i], 
            g_nccl.stream[i] ) );
    }
    bGroupOpen = false;
    nccl(GroupEnd() );
    return ncclSuccess;

Error:
    if ( bGroupOpen ) {
        ncclGroupEnd();
    }
    // A CUDA failure reaches here with ncclStatus still clear, so report
    // something a caller can print either way.
    return (ncclSuccess != ncclStatus) ? ncclStatus : ncclInternalError;
}

float
ComputeGravitation_multiGPU_2D( 
    float *force, 
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;
    ncclResult_t ncclStatus;

    float ret = 0.0f;
    int oldDevice;
    int cBlocks;
    size_t targetsPerRow;
    size_t sourcesPerCol;

    std::chrono::steady_clock::time_point start, end;

    if ( g_numGPUs > g_maxGPUs ) {
        return 0.0f;
    }
    if ( ! multiGPU2DAcquire( g_numGPUs ) ) {
        return 0.0f;
    }
    targetsPerRow = N / g_nccl.rows;
    sourcesPerCol = N / g_nccl.cols;
    if ( (0 != N % g_nccl.rows) || (0 != N % g_nccl.cols) ) {
        fprintf( stderr, 
            "2D: %zu bodies do not divide evenly over a %dx%d grid\n",
            N, g_nccl.rows, g_nccl.cols );
        return 0.0f;
    }
    // The tile kernel stages a whole block of threads' worth of sources at a
    // time, so a column's share of the bodies has to be a multiple of the
    // thread count.
    if ( 0 != sourcesPerCol % g_2DBlockSize ) {
        fprintf( stderr, 
            "2D: %zu sources per column is not a multiple of %d\n",
            sourcesPerCol, g_2DBlockSize );
        return 0.0f;
    }
    if ( ! multiGPUBuffersAcquire( g_numGPUs, N ) ) {
        return 0.0f;
    }
    cuda(GetDevice( &oldDevice ) );

    cBlocks = (int) ((targetsPerRow + g_2DBlockSize - 1) / g_2DBlockSize);
    if ( cBlocks > g_2DMaxBlocks ) {
        cBlocks = g_2DMaxBlocks;
    }

    start = std::chrono::steady_clock::now();

    // Each device uploads the two blocks its tile needs, into the positions
    // they occupy in the whole array, so the tile kernel can address them
    // without any rearrangement.  When the grid is square and the device is
    // on the diagonal the two blocks are the same one, and the second upload
    // is skipped.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        int row = i / g_nccl.cols;
        int col = i % g_nccl.cols;
        cuda(SetDevice( i ) );
        cuda(MemcpyAsync( 
            g_multiGPUBuffers[i].posMass + 4*targetsPerRow*row, 
            g_hostAOS_PosMass + 4*targetsPerRow*row, 
            4*targetsPerRow*sizeof(float), 
            cudaMemcpyHostToDevice, 
            g_nccl.stream[i] ) );
        if ( (size_t) row*targetsPerRow != (size_t) col*sourcesPerCol ||
             targetsPerRow != sourcesPerCol ) {
            cuda(MemcpyAsync( 
                g_multiGPUBuffers[i].posMass + 4*sourcesPerCol*col, 
                g_hostAOS_PosMass + 4*sourcesPerCol*col, 
                4*sourcesPerCol*sizeof(float), 
                cudaMemcpyHostToDevice, 
                g_nccl.stream[i] ) );
        }
        cuda(MemsetAsync( 
            g_multiGPUBuffers[i].force, 
            0, 
            3*targetsPerRow*sizeof(float), 
            g_nccl.stream[i] ) );
        ComputeNBodyGravitation_2D
            <<<cBlocks, g_2DBlockSize, g_2DBlockSize*sizeof(float4), 
               g_nccl.stream[i]>>>( 
            g_multiGPUBuffers[i].force,
            g_multiGPUBuffers[i].posMass,
            softeningSquared,
            targetsPerRow*row,
            targetsPerRow,
            sourcesPerCol*col,
            sourcesPerCol );
    }

    // Complete the forces by summing each row's partials.
    ncclStatus = reduceRowForces( 3*targetsPerRow );
    if ( ncclSuccess != ncclStatus ) {
        fprintf( stderr, "2D: row reduction failed: %s\n", 
            ncclGetErrorString( ncclStatus ) );
        goto Error;
    }

    // Every device in a row now holds that row's forces, so one device per
    // row returns them; the first column is as good a choice as any.
    for ( int i = 0; i < g_numGPUs; i++ ) {
        int row = i / g_nccl.cols;
        int col = i % g_nccl.cols;
        if ( 0 != col ) {
            continue;
        }
        cuda(SetDevice( i ) );
        cuda(MemcpyAsync( 
            g_hostAOS_Force + 3*targetsPerRow*row, 
            g_multiGPUBuffers[i].force, 
            3*targetsPerRow*sizeof(float), 
            cudaMemcpyDeviceToHost, 
            g_nccl.stream[i] ) );
    }
    for ( int i = 0; i < g_numGPUs; i++ ) {
        cuda(SetDevice( i ) );
        cuda(StreamSynchronize( g_nccl.stream[i] ) );
    }
    end = std::chrono::steady_clock::now();
    ret = std::chrono::duration<double>(end - start).count() * 1000.0f;

Error:
    cudaSetDevice( oldDevice );
    return ret;
}
