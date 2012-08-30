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

    CUresult Initialize( int ordinal, const list<string>& moduleList );
    CUresult loadModuleFromFile( const char *szFilename );

private:
    CUdevice m_device;
    CUcontext m_context;
    map<const char *, CUmodule> m_modules;
};

inline chCUDADevice::chCUDADevice()
{
    m_device = 0;
    m_context = 0;
}

CUresult
chCUDADevice::Initialize( int ordinal, const list<string>& moduleList )
{
    CUresult status;
    CUdevice device = 0;
    CUcontext ctx = 0;

    CUDA_CHECK( cuDeviceGet( &device, ordinal ) );
    CUDA_CHECK( cuCtxCreate( &ctx, 0, device ) );
    CUDA_CHECK( cuCtxPopCurrent( &ctx ) );

Error:
    return status;
}

vector<chCUDADevice *> g_CUDAdevices;

CUresult
chCUDAInitialize( const list<string>& moduleList )
{
    CUresult status;
    int cDevices;
    int cDevicesInitialized = 0;
    
    CUDA_CHECK( cuInit( 0 ) );
    CUDA_CHECK( cuDeviceGetCount( &cDevices ) );
    for ( int i = 0; i < cDevices; i++ ) {
        CUdevice device;
        CUcontext ctx = 0;
        chCUDADevice *newDevice = 0;

        newDevice = new chCUDADevice;
        if ( ! newDevice ) {
            status = CUDA_ERROR_OUT_OF_MEMORY;
            goto Error;
        }
        CUDA_CHECK( newDevice->Initialize( i, moduleList ) );
        g_CUDAdevices.push_back( newDevice );

    }
Error:
    for ( int i = 0; i < cDevices; i++ ) {
    }
    return status;
}

int
main( int argc, char *argv[] )
{
    CUresult status;

    list<string> moduleList;

    CUDA_CHECK( chCUDAInitialize( moduleList ) );
Error:
    return 1;
}
