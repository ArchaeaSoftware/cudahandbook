/*
 *
 * chDrv.h
 *
 * Header file for helper classes and functions for driver API.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
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

#ifndef __CHDRV_H__
#define __CHDRV_H__

#include <cuda.h>

#include <chDrv.h>

#include <chError.h>

#include <stdio.h>
#include <string>
#include <list>
#include <map>
#include <vector>

using namespace std;

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

extern vector<chCUDADevice *> g_CUDAdevices;

extern CUresult chCUDAInitialize( list<string>& moduleList );

#endif
