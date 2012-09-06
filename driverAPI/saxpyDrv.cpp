/*
 *
 * saxpyDrv.cpp
 *
 * Microdemo to illustrate how to initialize the driver API.
 *
 * Build with: nvcc -I ..\chLib <options> saxpyDrv.cpp chDrv.cpp
 * (You also can build with your local C compiler - gcc or cl.exe)
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2012, Archaea Software, LLC.
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
#include <stdlib.h> // for rand()

#include <chDrv.h>

#include <chError.h>
#include <chCommandLine.h>

CUresult
TestSAXPY( chCUDADevice *chDevice, size_t N, float alpha )
{
    CUresult status;
    CUdeviceptr dptrOut = 0;
    CUdeviceptr dptrIn = 0;
    float *hostOut = 0;
    float *hostIn = 0;

    CUDA_CHECK( cuCtxPushCurrent( chDevice->context() ) );

    CUDA_CHECK( cuMemAlloc( &dptrOut, N*sizeof(float) ) );
    CUDA_CHECK( cuMemsetD32( dptrOut, 0, N ) );
    CUDA_CHECK( cuMemAlloc( &dptrIn, N*sizeof(float) ) );
    CUDA_CHECK( cuMemHostAlloc( (void **) &hostOut, N*sizeof(float), 0 ) );
    CUDA_CHECK( cuMemHostAlloc( (void **) &hostIn, N*sizeof(float), 0 ) );
    for ( size_t i = 0; i < N; i++ ) {
        hostIn[i] = (float) rand() / (float) RAND_MAX;
    }
    CUDA_CHECK( cuMemcpyHtoDAsync( dptrIn, hostIn, N*sizeof(float ), NULL ) );

    {
        CUmodule moduleSAXPY;
        CUfunction kernelSAXPY;
        void *params[] = { &dptrOut, &dptrIn, &N, &alpha };
        
        moduleSAXPY = chDevice->module( "saxpy.ptx" );
        if ( ! moduleSAXPY ) {
            status = CUDA_ERROR_NOT_FOUND;
            goto Error;
        }
        CUDA_CHECK( cuModuleGetFunction( &kernelSAXPY, moduleSAXPY, "saxpy" ) );

        CUDA_CHECK( cuLaunchKernel( kernelSAXPY, 1500, 1, 1, 512, 1, 1, 0, NULL, params, NULL ) );

    }

    CUDA_CHECK( cuMemcpyDtoHAsync( hostOut, dptrOut, N*sizeof(float), NULL ) );
    CUDA_CHECK( cuCtxSynchronize() );
    for ( size_t i = 0; i < N; i++ ) {
        if ( fabsf( hostOut[i] - alpha*hostIn[i] ) > 1e-5f ) {
            status = CUDA_ERROR_UNKNOWN;
            goto Error;
        }
    }
    status = CUDA_SUCCESS;
    printf( "Well it worked!\n" );

Error:
    cuCtxPopCurrent( NULL );
    cuMemFreeHost( hostOut );
    cuMemFreeHost( hostIn );
    cuMemFree( dptrOut );
    cuMemFree( dptrIn );
    return status;
}

int
main( int argc, char *argv[] )
{
    CUresult status;

    list<string> moduleList;
    moduleList.push_back( "saxpy.ptx" );

    CUDA_CHECK( chCUDAInitialize( moduleList ) );
    for ( vector<chCUDADevice *>::iterator it  = g_CUDAdevices.begin();
                                           it != g_CUDAdevices.end();
                                           it++ ) {
        char deviceName[256];
        chCUDADevice *chDevice = *it;
        CUDA_CHECK( cuDeviceGetName( deviceName, 255, chDevice->device() ) );
        printf( "Testing SAXPY on %s (device %d)...\n", deviceName, chDevice->device() );
        CUDA_CHECK( TestSAXPY( chDevice, 16*1048576, 2.0 ) );
    }
    return 0;
Error:
    return 1;
}
