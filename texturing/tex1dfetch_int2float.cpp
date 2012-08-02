/*
 *
 * Copyright (C) 2011 by Archaea Software, LLC.  
 *      All rights reserved.
 *
 */

#include <cuda.h>

#include <stdio.h>

#include <chError.h>    

#define NUM_FLOATS 256

CUdevice g_device;
CUcontext g_ctx;
CUmodule g_module;
CUfunction g_function;
CUtexref g_texref;

float
TexPromoteToFloat( char c )
{
    if ( c == (char) 0x80 ) {
        return -1.0f;
    }
    return (float) c / 127.0f;
}

float
TexPromoteToFloat( short s )
{
    if ( s == (short) 0x8000 ) {
        return -1.0f;
    }
    return (float) s / 32767.0f;
}

float
TexPromoteToFloat( unsigned char uc )
{
    return (float) uc / 255.0f;
}

float
TexPromoteToFloat( unsigned short us )
{
    return (float) us / 65535.0f;
}

void
PrintTex( float *host, size_t N )
{
    CUdeviceptr device;
    CUresult status;
    memset( host, 0, N*sizeof(float) );
    CUDA_CHECK(cuMemHostGetDevicePointer( &device, host, 0 ));

    {
        int offset = 0;
        CUDA_CHECK(cuParamSetv(g_function, offset, &device, sizeof(CUdeviceptr))); offset += sizeof(CUdeviceptr);
        CUDA_CHECK(cuParamSetv(g_function, offset, &N, sizeof(size_t))); offset += sizeof(size_t);
        CUDA_CHECK(cuParamSetSize(g_function, offset ));
    }
    CUDA_CHECK(cuFuncSetBlockShape(g_function, 2, 1, 1));
    CUDA_CHECK(cuLaunchGrid(g_function, 384, 1));
    CUDA_CHECK(cuCtxSynchronize());
    for ( int i = 0; i < N; i++ ) {
        char c = (char) i;
        printf( "%.2f ", host[i] );
        if ( host[i] != TexPromoteToFloat( c ) )
            _asm int 3
/*        
// this works for unsigned char
        if ( fabsf(host[i] - (float)i*(1.0f/255.0f) ) > 1e-5f )
            _asm int 3
*/
    }
    printf( "\n" );
Error:;
}

static CUresult
LoadPTXModule( CUmodule *mod, const char *filename )
{
    FILE *f = fopen( filename, "rb" );
    long size;
    void *p = 0;
    CUresult status;
    if ( ! f ) {
        return CUDA_ERROR_NOT_FOUND;
    }
    fseek( f, 0, SEEK_END );
    size = ftell( f );
    fseek( f, 0, SEEK_SET );
    p = malloc( size+1 );
    if ( ! p ) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if ( 1 != fread( p, size, 1, f ) ) {
        status = CUDA_ERROR_UNKNOWN;
        goto Error;
    }
    ((char *) p)[size] = '\0';
    status = cuModuleLoadDataEx( mod, p, 0, NULL, NULL );
Error:
    fclose( f );
    free( p );
    return status;
}

int
main( int argc, char *argv[] )
{
    CUdeviceptr p = 0;
    unsigned char *finHost;
    CUdeviceptr finDevice;

    float *foutHost;
    CUdeviceptr foutDevice;
    CUresult status;

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGet(&g_device, 0));
    CUDA_CHECK(cuCtxCreate(&g_ctx, CU_CTX_MAP_HOST, g_device));
    CUDA_CHECK(LoadPTXModule(&g_module, "tex1dfetch_int2float_kernel.ptx"));
    CUDA_CHECK(cuModuleGetFunction(&g_function, g_module, "TexReadout"));
    CUDA_CHECK(cuModuleGetTexRef(&g_texref, g_module, "tex1"));

    CUDA_CHECK(cuMemAlloc( &p, NUM_FLOATS*sizeof(float)) );
    CUDA_CHECK(cuMemHostAlloc( (void **) &finHost, NUM_FLOATS*sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostGetDevicePointer( &finDevice, finHost, 0 ));

    CUDA_CHECK(cuTexRefSetFormat(g_texref, CU_AD_FORMAT_SIGNED_INT8, 1));
    //CUDA_CHECK(cuTexRefSetFlags(g_texref, CU_TRSF_READ_AS_INTEGER));
    CUDA_CHECK(cuTexRefSetAddress(NULL, g_texref, finDevice, NUM_FLOATS*sizeof(float)));

    CUDA_CHECK(cuMemHostAlloc( (void **) &foutHost, NUM_FLOATS*sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostGetDevicePointer( &foutDevice, foutHost, 0 ));

    for ( int i = 0; i < NUM_FLOATS; i++ ) {
        finHost[i] = (unsigned char) i;//(float) i;
    }
    PrintTex( foutHost, NUM_FLOATS );

Error:
    cuMemFree( p );
}
