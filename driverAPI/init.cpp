/*
 *
 * initDrv.cpp
 *
 * Microdemo to illustrate how to initialize the driver API.
 *
 * Build with: nvcc -I ..\chLib <options> initDrv.cpp
 * (You also can build with your local C compiler - gcc or cl.exe)
 * Requires: No minimum SM requirement.
 *
  * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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

#include <cuda.h>

#include <chError.h>
#include <chCommandLine.h>

using namespace std;

#include <string>
#include <list>
#include <map>
#include <vector>


class chCUDADevice
{
public:
    chCUDADevice();
    virtual ~chCUDADevice();

    CUresult Initialize( 
        int ordinal, 
        list<string>& moduleList,
        unsigned int Flags = 0,
        unsigned int numOptions = 0,
        CUjit_option *options = NULL,
        void **optionValues = NULL );
    CUresult loadModuleFromFile( 
        CUmodule *pModule,
        string fileName,
        unsigned int numOptions = 0,
        CUjit_option *options = NULL,
        void **optionValues = NULL );

    CUdevice device() const { return m_device; }
    CUcontext context() const { return m_context; }
    CUmodule module( string s ) const { return (*m_modules.find(s)).second; }

private:
    CUdevice m_device;
    CUcontext m_context;
    map<string, CUmodule> m_modules;

};

inline
chCUDADevice::chCUDADevice()
{
    m_device = 0;
    m_context = 0;
}

inline 
chCUDADevice::~chCUDADevice()
{
    for ( map<string, CUmodule>::iterator it = m_modules.begin();
          it != m_modules.end();
          it ++ ) {
        cuModuleUnload( (*it).second );
    }
    cuCtxDestroy( m_context );
}

CUresult
chCUDADevice::loadModuleFromFile( 
    CUmodule *pModule,
    string fileName,
    unsigned int numOptions,
    CUjit_option *options,
    void **optionValues )
{
    CUresult status;
    CUmodule module = 0;
    long int lenFile;
    FILE *file = fopen( fileName.c_str(), "rb" );
    char *fileContents = 0;
    if ( ! file ) {
        status = CUDA_ERROR_NOT_FOUND;
        goto Error;
    }
    if ( 0 != fseek( file, 0, SEEK_END ) ) {
        fclose( file );
        status = CUDA_ERROR_UNKNOWN;
        goto Error;
    }
    lenFile = ftell( file );
    fileContents = (char *) malloc( lenFile+1 );
    if ( fileContents ) {
        fseek( file, 0, SEEK_SET );
        if ( lenFile != fread( fileContents, 1, lenFile, file ) ) {
            fclose( file );
            status = CUDA_ERROR_UNKNOWN;
            goto Error;
        }
        fileContents[lenFile] = '\0'; // NULL terminate the string
        status = cuModuleLoadDataEx( &module, fileContents, numOptions, options, optionValues );
        free( fileContents );
        if ( status != CUDA_SUCCESS )
            goto Error;
        m_modules.insert( pair<string, CUmodule>(fileName, module) );
    }
Error:
    return status;
}

CUresult
chCUDADevice::Initialize( 
    int ordinal, 
    list<string>& moduleList,
    unsigned int CtxCreateFlags,
    unsigned int numOptions,
    CUjit_option *options,
    void **optionValues )
{
    CUresult status;
    CUdevice device;
    CUcontext ctx = 0;

    CUDA_CHECK( cuDeviceGet( &device, ordinal ) );
    CUDA_CHECK( cuCtxCreate( &ctx, CtxCreateFlags, device ) );
    for ( list<string>::iterator it  = moduleList.begin();
                                 it != moduleList.end();
                                 it++ ) {
        CUDA_CHECK( loadModuleFromFile( NULL, *it, numOptions, options, optionValues ) );
    }
    CUDA_CHECK( cuCtxPopCurrent( &ctx ) );
    m_context = ctx;
    return CUDA_SUCCESS;
Error:
    cuCtxDestroy( ctx );
    return status;
}

vector<chCUDADevice *> g_CUDAdevices;

CUresult
chCUDAInitialize( list<string>& moduleList )
{
    CUresult status;
    int cDevices;
    int cDevicesInitialized = 0;
    chCUDADevice *newDevice;
    
    CUDA_CHECK( cuInit( 0 ) );
    CUDA_CHECK( cuDeviceGetCount( &cDevices ) );
    for ( int i = 0; i < cDevices; i++ ) {
        CUdevice device;
        CUcontext ctx = 0;
        newDevice = 0;

        newDevice = new chCUDADevice;
        if ( ! newDevice ) {
            status = CUDA_ERROR_OUT_OF_MEMORY;
            goto Error;
        }

        CUDA_CHECK( newDevice->Initialize( i, moduleList ) );
        g_CUDAdevices.push_back( newDevice );
    }
    return CUDA_SUCCESS;
Error:
    while ( ! g_CUDAdevices.empty() ) {
        delete (*g_CUDAdevices.end());
        g_CUDAdevices.pop_back();
    }
    delete newDevice;
    return status;
}

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
