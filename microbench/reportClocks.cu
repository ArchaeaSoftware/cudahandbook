/*
 *
 * reportClocks.cu
 *
 * Reports how grid and block IDs are assigned by the hardware.
 *
 * Build with: nvcc -I ../chLib <options> reportClocks.cu
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

__global__ void
WriteClockValues( unsigned int *completionTimes, unsigned int *threadIDs )
{
    size_t globalBlock = blockIdx.x+blockDim.x*(blockIdx.y+blockDim.y*blockIdx.z);
    size_t globalThread = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    
    size_t totalBlockSize = blockDim.x*blockDim.y*blockDim.z;
    size_t globalIndex = globalBlock*totalBlockSize + globalThread;

    completionTimes[globalIndex] = clock();
    threadIDs[globalIndex] = threadIdx.y<<4|threadIdx.x;
}

void
WriteOutput( FILE *out, dim3 gridSize, dim3 blockSize, const unsigned int *hostOut )
{
    int index = 0;
    for ( int gridZ = 0; gridZ < gridSize.z; gridZ++ ) {
        for ( int gridY = 0; gridY < gridSize.y; gridY++ ) {
            for ( int gridX = 0; gridX < gridSize.x; gridX++ ) {
                for ( int threadZ = 0; threadZ < blockSize.z; threadZ++ ) {
                    printf( "Grid (%d, %d, %d) - slice %d:\n", gridX, gridY, gridZ, threadZ );
                    for ( int threadY = 0; threadY < blockSize.y; threadY++ ) {
                        for ( int threadX = 0; threadX < blockSize.x; threadX++ ) {
                            fprintf( out, "%4x", hostOut[index] );
                            index++;
                        }
                        fprintf(out, "\n" );
                    }
                }
            }
        }
    }
}

bool
ReportTimesAndIDs( FILE *clocksFile, FILE *tidsFile, dim3 gridSize, dim3 blockSize )
{
    cudaError_t status;
    bool ret = false;
    size_t totalBlockSize = blockSize.x*blockSize.y*blockSize.z;
    size_t numTimes = totalBlockSize*gridSize.x*gridSize.y*gridSize.z;
    cudaEvent_t start = 0;
    cudaEvent_t stop = 0;
    
    unsigned int *deviceClockValues = 0;
    unsigned int *deviceThreadIDs = 0;
    unsigned int *hostOut = 0;

    cuda(Malloc( &deviceClockValues, numTimes*sizeof(int) ) );
    cuda(Malloc( &deviceThreadIDs, numTimes*sizeof(int) ) );
    cuda(MallocHost( &hostOut, numTimes*sizeof(int) ) );

    cuda(EventCreate( &start ) );
    cuda(EventCreate( &stop ) );

    WriteClockValues<<<gridSize, blockSize>>>( deviceClockValues, deviceThreadIDs );
    cuda(DeviceSynchronize() );

    cuda(EventRecord( start, 0 ) );
    WriteClockValues<<<gridSize, blockSize>>>( deviceClockValues, deviceThreadIDs );
    cuda(EventRecord( stop, 0 ) );

    cuda(DeviceSynchronize() );

    {
        float ms;
        cuda(EventElapsedTime( &ms, start, stop ) );
        printf( "%.2f ms for %d threads = %.2f us/thread\n", ms, (int) numTimes, ms*1e3/numTimes );
    }

    if ( clocksFile ) {
        cuda(Memcpy( hostOut, deviceClockValues, numTimes*sizeof(int), cudaMemcpyDeviceToHost ) );
        // turn clock values into completion times by subtracting minimum reported clock value
        unsigned int minTime = ~(unsigned int) 0;
        for ( size_t i = 0; i < numTimes; i++ ) {
            if ( hostOut[i] < minTime ) {
                minTime = hostOut[i];
            }
        }
        for ( size_t i = 0; i < numTimes; i++ ) {
            hostOut[i] -= minTime;
        }
        fprintf( clocksFile, "Completion times (clocks):\n" );
        WriteOutput( clocksFile, gridSize, blockSize, hostOut );
    }
    if ( tidsFile ) {
        cuda(Memcpy( hostOut, deviceThreadIDs, numTimes*sizeof(int), cudaMemcpyDeviceToHost ) );
        fprintf( tidsFile, "Thread IDs:\n" );
        WriteOutput( tidsFile, gridSize, blockSize, hostOut );
    }

    ret = true;
Error:
    cudaFree( deviceClockValues );
    cudaFree( deviceThreadIDs );
    cudaFreeHost( hostOut );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return ret;
}

int
main( int argc, char *argv[] )
{
    dim3 blockSize;
    dim3 gridSize;

    blockSize.x = blockSize.y = blockSize.z = 1;
    gridSize.x = gridSize.y = gridSize.z = 1;

    gridSize.x = 2;

    blockSize.x = 16;
    blockSize.y = 8;
    ReportTimesAndIDs( stdout, stdout, gridSize, blockSize );

    blockSize.x = 14;
    blockSize.y = 8;
    ReportTimesAndIDs( stdout, stdout, gridSize, blockSize );

#if 0
    for ( gridSize.x = 1; gridSize.x <= 8; gridSize.x++ ) {

        blockSize.x = blockSize.y = 16;

        ReportTimesAndIDs( stdout, NULL, gridSize, blockSize );
        
        blockSize.x = 17;
        blockSize.y = 16;
        ReportTimesAndIDs( stdout, NULL, gridSize, blockSize );

        blockSize.x = 18;
        blockSize.y = 16;
        ReportTimesAndIDs( stdout, NULL, gridSize, blockSize );
    }
#endif
}
