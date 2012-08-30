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
        unsigned int numOptions = 0,
        CUjit_option *options = NULL,
        void **optionValues = NULL );
    CUresult loadModuleFromFile( 
        CUmodule *pModule,
        const char *fileName,
        unsigned int numOptions = 0,
        CUjit_option *options = NULL,
        void **optionValues = NULL );

private:
    CUdevice m_device;
    CUcontext m_context;
    map<const char *, CUmodule> m_modules;

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
    for ( map<const char *, CUmodule>::iterator it = m_modules.begin();
          it != m_modules.end();
          it ++ ) {
        cuModuleUnload( (*it).second );
    }
    cuCtxDestroy( m_context );
}

CUresult
chCUDADevice::loadModuleFromFile( 
    CUmodule *pModule,
    const char *fileName,
    unsigned int numOptions,
    CUjit_option *options,
    void **optionValues )
{
    CUresult status;
    CUmodule module = 0;
    long int lenFile;
    FILE *file = fopen( fileName, "rb" );
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
        m_modules.insert( pair<const char *, CUmodule>(fileName, module) );
    }
Error:
    return status;
}

CUresult
chCUDADevice::Initialize( 
    int ordinal, 
    list<string>& moduleList,
    unsigned int numOptions,
    CUjit_option *options,
    void **optionValues )
{
    CUresult status;
    CUdevice device;
    CUcontext ctx = 0;

    CUDA_CHECK( cuDeviceGet( &device, ordinal ) );
    CUDA_CHECK( cuCtxCreate( &ctx, 0, device ) );
    for ( list<string>::iterator it  = moduleList.begin();
                                 it != moduleList.end();
                                 it++ ) {
        CUDA_CHECK( loadModuleFromFile( NULL, (*it).c_str(), numOptions, options, optionValues ) );
    }
    CUDA_CHECK( cuCtxPopCurrent( &ctx ) );
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

int
main( int argc, char *argv[] )
{
    CUresult status;

    list<string> moduleList;
    moduleList.push_back( "saxpy.ptx" );

    CUDA_CHECK( chCUDAInitialize( moduleList ) );
Error:
    return 1;
}
