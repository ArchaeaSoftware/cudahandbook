/*
 *
 * histogramNPP.cuh
 *
 * Implementation of histogram that uses the NVIDIA Performance Primitives.
 *
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2013, Archaea Software, LLC.
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

#include <npp.h>

void
GPUhistogramNPP(
    float *ms,
    unsigned int *pHist,
    const unsigned char *dptrBase, size_t dptrPitch,
    int x, int y,
    int w, int h, 
    dim3 threads )
{
    cudaError_t status;
    Npp8u *pDeviceBuffer = 0;
    NppiSize oSizeROI = {w, h};

    const int binCount = 256;
    const int levelCount = binCount+1;

    cudaEvent_t start = 0, stop = 0;

    cuda(EventCreate( &start, 0 ) );
    cuda(EventCreate( &stop, 0 ) );

    // create device scratch buffer for nppiHistogram
    int nDeviceBufferSize;
    nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount ,&nDeviceBufferSize);
    cuda(Malloc((void **)&pDeviceBuffer, nDeviceBufferSize) );

    cuda(EventRecord( start, 0 ) );

    // compute the histogram
    if ( NPP_NO_ERROR != nppiHistogramEven_8u_C1R(
        (const Npp8u *) dptrBase, 
        dptrPitch, 
        oSizeROI,
        (Npp32s *) pHist, 
        levelCount, 0, binCount,
        pDeviceBuffer ) )
        goto Error;
    cuda(EventRecord( stop, 0 ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( ms, start, stop ) );

Error:
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaFree( pDeviceBuffer );
    return;
}

