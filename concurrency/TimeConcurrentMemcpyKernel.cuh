/*
 *
 * TimeConcurrentMemcpyKernel.cuh
 *
 * CUDA header to implement timing of software-pipelined download/
 * launch/upload operations in a specified number of streams.
 *
 * Included by:
 *     concurrencyKernelMapped.cu
 *     concurrencyMemcpyKernel.cu
 *     concurrencyMemcpyKernelMapped.cu
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

#ifndef __CUDAHANDBOOK_TIMECONCURRENTMEMCPYKERNEL_CUH__
#define __CUDAHANDBOOK_TIMECONCURRENTMEMCPYKERNEL_CUH__

#ifndef __CUDAHANDBOOK__ADD_KERNEL__
#include "AddKernel.cuh"
#endif

//
// Times the operation using the specified input size and 
// number of streams.
//

bool
TimeConcurrentMemcpyKernel( 
    float *times, size_t N, 
    const chShmooRange& cyclesRange,
    const chShmooRange& streamsRange, 
    int numBlocks )
{
    cudaError_t status;
    float ret = 0.0f;
    int *hostIn = 0;
    int *hostOut = 0;
    int *deviceIn = 0;
    int *deviceOut = 0;

    const int maxStreams = streamsRange.max();
    cudaStream_t *streams = 0;

    size_t intsPerStream = N / streamsRange.max();

    const int numEvents = 2;
    cudaEvent_t events[numEvents];

    memset( events, 0, sizeof(events) );

    for ( int i = 0; i < numEvents; i++ ) {
        events[i] = NULL;
        cuda(EventCreate( &events[i] ) );
    }
    streams = (cudaStream_t *) malloc( maxStreams*sizeof(cudaStream_t) );
    if ( ! streams )
        goto Error;
    memset( streams, 0, maxStreams*sizeof(cudaStream_t) );
    for ( int i = 0; i < maxStreams; i++ ) {
        cuda(StreamCreate( &streams[i] ) );
    }

    cuda(MallocHost( &hostIn, N*sizeof(int) ) );
    cuda(MallocHost( &hostOut, N*sizeof(int) ) );
    cuda(Malloc( &deviceIn, N*sizeof(int) ) );
    cuda(Malloc( &deviceOut, N*sizeof(int) ) );

    for ( size_t i = 0; i < N; i++ ) {
        hostIn[i] = rand();
    }

    cuda(DeviceSynchronize() );

    for ( chShmooIterator streamCount(streamsRange); streamCount; streamCount++ ) {
        int numStreams = *streamCount;

        for ( chShmooIterator cycles(cyclesRange); cycles; cycles++ ) {
            size_t intsLeft;

            printf( "." ); fflush( stdout );

            cuda(EventRecord( events[0], NULL ) );

            intsLeft = N;
            for ( int stream = 0; stream < numStreams; stream++ ) {
                size_t intsToDo = (intsLeft < intsPerStream) ? intsLeft : intsPerStream;
                cuda(MemcpyAsync( 
                    deviceIn+stream*intsPerStream, 
                    hostIn+stream*intsPerStream, 
                    intsToDo*sizeof(int), 
                    cudaMemcpyHostToDevice, streams[stream] ) );
                intsLeft -= intsToDo;
            }

            intsLeft = N;
            for ( int stream = 0; stream < numStreams; stream++ ) {
                size_t intsToDo = (intsLeft < intsPerStream) ? intsLeft : intsPerStream;
                AddKernel<<<numBlocks, 256, 0, streams[stream]>>>( 
                    deviceOut+stream*intsPerStream, 
                    deviceIn+stream*intsPerStream, 
                    intsToDo, 0xcc, *cycles );
                intsLeft -= intsToDo;
            }

            intsLeft = N;
            for ( int stream = 0; stream < numStreams; stream++ ) {
                size_t intsToDo = (intsLeft < intsPerStream) ? intsLeft : intsPerStream;
                cuda(MemcpyAsync( 
                    hostOut+stream*intsPerStream, 
                    deviceOut+stream*intsPerStream, 
                    intsToDo*sizeof(int), 
                    cudaMemcpyDeviceToHost, streams[stream] ) );
                intsLeft -= intsToDo;
            }

            cuda(EventRecord( events[1], NULL ) );
            cuda(DeviceSynchronize() );

            // confirm that the computation was done correctly
            for ( size_t i = 0; i < N; i++ ) {
                CH_ASSERT( hostOut[i] == hostIn[i]+*cycles*0xcc );
                if ( hostOut[i] != hostIn[i]+*cycles*0xcc ) {
                    return false;
                }
            }

            cuda(EventElapsedTime( times, events[0], events[1] ) );

            times += 1;
        }
    }

    ret = true;

Error:
    for ( int i = 0; i < numEvents; i++ ) {
        cudaEventDestroy( events[i] );
    }
    for ( int i = 0; i < maxStreams; i++ ) {
        cudaStreamDestroy( streams[i] );
    }
    free( streams );

    cudaFree( deviceIn );
    cudaFree( deviceOut );
    cudaFreeHost( hostOut );
    cudaFreeHost( hostIn );
    return ret;
}

#endif
