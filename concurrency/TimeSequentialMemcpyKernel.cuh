/*
 *
 * TimeSequentialMemcpyKernel.cuh
 *
 * CUDA header to implement timing of sequential upload/launch/download
 * operations.
 *
 * Included by concurrencyMemcpyKernel.cu
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

#ifndef __CUDAHANDBOOK_TIMESEQUENTIALMEMCPYKERNEL_CUH__
#define __CUDAHANDBOOK_TIMESEQUENTIALMEMCPYKERNEL_CUH__

bool
TimeSequentialMemcpyKernel( 
    float *timesHtoD, float *timesKernel, float *timesDtoH, float *timesTotal,
    size_t N, 
    const chShmooRange& cyclesRange,
    int numBlocks )
{
    cudaError_t status;
    bool ret = false;
    int *hostIn = 0;
    int *hostOut = 0;
    int *deviceIn = 0;
    int *deviceOut = 0;
    const int numEvents = 4;
    cudaEvent_t events[numEvents];

    for ( int i = 0; i < numEvents; i++ ) {
        events[i] = NULL;
        cuda(EventCreate( &events[i] ) );
    }
    cuda(MallocHost( &hostIn, N*sizeof(int) ) );
    cuda(MallocHost( &hostOut, N*sizeof(int) ) );
    cuda(Malloc( &deviceIn, N*sizeof(int) ) );
    cuda(Malloc( &deviceOut, N*sizeof(int) ) );

    for ( size_t i = 0; i < N; i++ ) {
        hostIn[i] = rand();
    }

    cuda(DeviceSynchronize() );

    for ( chShmooIterator cycles(cyclesRange); cycles; cycles++ ) {

        printf( "." ); fflush( stdout );

        cuda(EventRecord( events[0], NULL ) );
        cuda(MemcpyAsync( deviceIn, hostIn, N*sizeof(int), 
            cudaMemcpyHostToDevice, NULL ) );
        cuda(EventRecord( events[1], NULL ) );
        AddKernel<<<numBlocks, 256>>>( deviceOut, deviceIn, N, 0xcc, *cycles );
        cuda(EventRecord( events[2], NULL ) );
        cuda(MemcpyAsync( hostOut, deviceOut, N*sizeof(int), 
            cudaMemcpyDeviceToHost, NULL ) );
        cuda(EventRecord( events[3], NULL ) );

        cuda(DeviceSynchronize() );

        cuda(EventElapsedTime( timesHtoD, events[0], events[1] ) );
        cuda(EventElapsedTime( timesKernel, events[1], events[2] ) );
        cuda(EventElapsedTime( timesDtoH, events[2], events[3] ) );
        cuda(EventElapsedTime( timesTotal, events[0], events[3] ) );

        timesHtoD += 1;
        timesKernel += 1;
        timesDtoH += 1;
        timesTotal += 1;
    }

    ret = true;

Error:
    for ( int i = 0; i < numEvents; i++ ) {
        cudaEventDestroy( events[i] );
    }

    cudaFree( deviceIn );
    cudaFree( deviceOut );
    cudaFreeHost( hostOut );
    cudaFreeHost( hostIn );
    return ret;
}

#endif
