/*
 *
 * chDrv.cpp
 *
 * Implementation file for helper classes and functions for driver API.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
 * All rights reserved.

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

#include <chDrv.h>

vector<chCUDADevice *> g_CUDAdevices;

CUresult
chCUDADevice::loadModuleFromFile( 
    CUmodule *pModule,
    string fileName,
    unsigned int numOptions,
    CUjit_option *options,
    void **optionValues )
{
    CUresult status_cuda;
    CUmodule module = 0;
    long int lenFile;
    FILE *file = fopen( fileName.c_str(), "rb" );
    char *fileContents = 0;
    if ( ! file ) {
        status_cuda = CUDA_ERROR_NOT_FOUND;
        goto Error_cuda;
    }
    if ( 0 != fseek( file, 0, SEEK_END ) ) {
        fclose( file );
        status_cuda = CUDA_ERROR_UNKNOWN;
        goto Error_cuda;
    }
    lenFile = ftell( file );
    fileContents = (char *) malloc( lenFile+1 );
    if ( fileContents ) {
        fseek( file, 0, SEEK_SET );
        if ( lenFile != fread( fileContents, 1, lenFile, file ) ) {
            fclose( file );
            status_cuda = CUDA_ERROR_UNKNOWN;
            goto Error_cuda;
        }
        fileContents[lenFile] = '\0'; // NULL terminate the string
        status_cuda = cuModuleLoadDataEx( &module, fileContents, numOptions, options, optionValues );
        free( fileContents );
        if ( status_cuda != CUDA_SUCCESS )
            goto Error_cuda;
        m_modules.insert( pair<string, CUmodule>(fileName, module) );
    }
Error_cuda:
    return status_cuda;
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
    CUresult status_cuda;
    CUdevice device;
    CUcontext ctx = 0;

    cu(DeviceGet( &device, ordinal ) );
    cu(CtxCreate( &ctx, CtxCreateFlags, device ) );
    for ( list<string>::iterator it  = moduleList.begin();
                                 it != moduleList.end();
                                 it++ ) {
        CUDA_CHECK( loadModuleFromFile( NULL, *it, numOptions, options, optionValues ) );
    }
    cu(CtxPopCurrent( &ctx ) );
    m_device = device;
    m_context = ctx;
    return CUDA_SUCCESS;
Error_cuda:
    cuCtxDestroy( ctx );
    return status_cuda;
}

CUresult
chCUDAInitialize( list<string>& moduleList )
{
    CUresult status_cuda;
    int cDevices;
    int cDevicesInitialized = 0;
    chCUDADevice *newDevice;
    
    cu(Init( 0 ) );
    cu(DeviceGetCount( &cDevices ) );
    for ( int i = 0; i < cDevices; i++ ) {
        CUdevice device;
        CUcontext ctx = 0;
        newDevice = 0;

        newDevice = new chCUDADevice;
        if ( ! newDevice ) {
            status_cuda = CUDA_ERROR_OUT_OF_MEMORY;
            goto Error_cuda;
        }

        CUDA_CHECK( newDevice->Initialize( i, moduleList ) );
        g_CUDAdevices.push_back( newDevice );
    }
    return CUDA_SUCCESS;
Error_cuda:
    while ( ! g_CUDAdevices.empty() ) {
        delete (*g_CUDAdevices.end());
        g_CUDAdevices.pop_back();
    }
    delete newDevice;
    return status_cuda;
}

