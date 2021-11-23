/*
 *
 * nbody.cu
 *
 * N-body example that illustrates gravitational simulation.
 * This is the type of computation that GPUs excel at:
 * parallelizable, with lots of FLOPS per unit of external 
 * memory bandwidth required.
 *
 * Build with: nvcc -I ../chLib nbody.cu nbody_CPU_SSE.cpp nbody_CPU_AOS.cpp nbody_CPU_AOS_tiled.cpp nbody_CPU_SSE_threaded.cpp nbody_CPU_SOA.cpp nbody_GPU_shared.cu nbody_multiGPU.cu nbody_multiGPU_threaded.cu
 *   On Linux: nvcc -I ../chLib nbody.cu nbody_CPU_SSE.cpp nbody_CPU_AOS.cpp nbody_CPU_AOS_tiled.cpp nbody_CPU_SSE_threaded.cpp nbody_CPU_SOA.cpp nbody_GPU_shared.cu nbody_multiGPU.cu nbody_multiGPU_threaded.cu -lcudart_static -ldl -lrt
 * Requires: No minimum SM requirement.  If SM 3.x is not available,
 * this application quietly replaces the shuffle and fast-atomic
 * implementations with the shared memory implementation.
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

#include <stdio.h>

// for kbhit()
#include <ch_conio.h>

#include <math.h>

#include <chCommandLine.h>
#include <chError.h>
#include <chThread.h>
#include <chTimer.h>

#include "nbody.h"

#include "bodybodyInteraction.cuh"

using namespace cudahandbook::threading;

inline void
randomVector( float v[3] )
{
    float lenSqr;
    do {
        v[0] = rand() / (float) RAND_MAX * 2 - 1;
        v[1] = rand() / (float) RAND_MAX * 2 - 1;
        v[2] = rand() / (float) RAND_MAX * 2 - 1;
        lenSqr = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
    } while ( lenSqr > 1.0f );
}

void
randomUnitBodies( float *pos, float *vel, size_t N )
{
    for ( size_t i = 0; i < N; i++ ) {
        randomVector( &pos[4*i] );
        randomVector( &vel[4*i] );
        pos[4*i+3] = 1.0f;  // unit mass
        vel[4*i+3] = 1.0f;
    }
}

template<typename T>
static float
relError( float a, float b )
{
    if ( a == b ) return 0.0f;
    return fabsf(a-b)/b;
}

bool g_bCUDAPresent;
bool g_bSM30Present;

float *g_hostAOS_PosMass;
float *g_hostAOS_VelInvMass;
float *g_hostAOS_Force;
float *g_hostAOS_gpuCrossCheckForce[32];

float *g_dptrAOS_PosMass;
float *g_dptrAOS_Force;

//
// threshold for soft comparisons when validating
// that forces add up to 0.
//
double g_ZeroThreshold;

bool g_bGPUTest;

// Buffer to hold the golden version of the forces, used for comparison
// Along with timing results, we report the maximum relative error with 
// respect to this array.
float *g_hostAOS_Force_Golden;

float *g_hostSOA_Pos[3];
float *g_hostSOA_Force[3];
float *g_hostSOA_Mass;
float *g_hostSOA_InvMass;

size_t g_N;

float g_softening = 0.1f;
float g_damping = 0.995f;
float g_dt = 0.016f;

template<typename T>
static T
relError( T a, T b )
{
    if ( a == b ) return 0.0f;
    T relErr = (a-b)/b;
    // Manually take absolute value
    return (relErr<0.0f) ? -relErr : relErr;
}

#include "nbody_CPU_AOS.h"
#include "nbody_CPU_AOS_tiled.h"
#include "nbody_CPU_SOA.h"
#include "nbody_CPU_SIMD.h"

#ifndef NO_CUDA
#include "nbody_GPU_AOS.cuh"
#include "nbody_GPU_AOS_const.cuh"
#include "nbody_GPU_AOS_tiled.cuh"
#include "nbody_GPU_AOS_tiled_const.cuh"
//#include "nbody_GPU_SOA_tiled.cuh"
#include "nbody_GPU_Shuffle.cuh"
#include "nbody_GPU_Atomic.cuh"
#endif

#include "kahan.h"

void
integrateGravitation_AOS( float *ppos, float *pvel, float *pforce, float dt, float damping, size_t N )
{
    for ( size_t i = 0; i < N; i++ ) {
        int index = 4*i;
        int indexForce = 3*i;

        float pos[3], vel[3], force[3];
        pos[0] = ppos[index+0];
        pos[1] = ppos[index+1];
        pos[2] = ppos[index+2];
        float invMass = pvel[index+3];

        vel[0] = pvel[index+0];
        vel[1] = pvel[index+1];
        vel[2] = pvel[index+2];

        force[0] = pforce[indexForce+0];
        force[1] = pforce[indexForce+1];
        force[2] = pforce[indexForce+2];

        // acceleration = force / mass;
        // new velocity = old velocity + acceleration * deltaTime
        vel[0] += (force[0] * invMass) * dt;
        vel[1] += (force[1] * invMass) * dt;
        vel[2] += (force[2] * invMass) * dt;

        vel[0] *= damping;
        vel[1] *= damping;
        vel[2] *= damping;

        // new position = old position + velocity * deltaTime
        pos[0] += vel[0] * dt;
        pos[1] += vel[1] * dt;
        pos[2] += vel[2] * dt;

        ppos[index+0] = pos[0];
        ppos[index+1] = pos[1];
        ppos[index+2] = pos[2];

        pvel[index+0] = vel[0];
        pvel[index+1] = vel[1];
        pvel[index+2] = vel[2];
    }
}

enum nbodyAlgorithm_enum g_Algorithm;

//
// g_maxAlgorithm is used to determine when to rotate g_Algorithm back to CPU_AOS
// If CUDA is present, it is CPU_SIMD_threaded, otherwise it depends on SM version
//
// The shuffle and tiled implementations are SM 3.0 only.
//
// The CPU and GPU algorithms must be contiguous, and the logic in main() to
// initialize this value must be modified if any new algorithms are added.
//
enum nbodyAlgorithm_enum g_maxAlgorithm;
bool g_bCrossCheck = true;
bool g_bUseSIMDForCrossCheck = true;
bool g_bNoCPU = false;
bool g_bGPUCrossCheck = false;
bool g_bGPUCrossCheckFile = false;
FILE *g_fGPUCrosscheckInput;
FILE *g_fGPUCrosscheckOutput;

bool
ComputeGravitation( 
    float *ms,
    float *maxRelError,
    nbodyAlgorithm_enum algorithm, 
    bool bCrossCheck )
{
    cudaError_t status;
    bool bSOA = false;

    // AOS -> SOA data structures in case we are measuring SOA performance
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostSOA_Pos[0][i]  = g_hostAOS_PosMass[4*i+0];
        g_hostSOA_Pos[1][i]  = g_hostAOS_PosMass[4*i+1];
        g_hostSOA_Pos[2][i]  = g_hostAOS_PosMass[4*i+2];
        g_hostSOA_Mass[i]    = g_hostAOS_PosMass[4*i+3];
        g_hostSOA_InvMass[i] = 1.0f / g_hostSOA_Mass[i];
    }

    if ( bCrossCheck ) {
#ifdef HAVE_SIMD_THREADED
        if ( g_bUseSIMDForCrossCheck ) {
            ComputeGravitation_SIMD_threaded(
                            g_hostSOA_Force,
                            g_hostSOA_Pos,
                            g_hostSOA_Mass,
                            g_softening*g_softening,
                            g_N );
            for ( size_t i = 0; i < g_N; i++ ) {
                g_hostAOS_Force_Golden[3*i+0] = g_hostSOA_Force[0][i];
                g_hostAOS_Force_Golden[3*i+1] = g_hostSOA_Force[1][i];
                g_hostAOS_Force_Golden[3*i+2] = g_hostSOA_Force[2][i];
            }
        }
        else {
#endif
            ComputeGravitation_AOS( 
                g_hostAOS_Force_Golden,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
#ifdef HAVE_SIMD_THREADED
        }
#endif
    }

    // CPU->GPU copies in case we are measuring GPU performance
    if ( g_bCUDAPresent ) {
        cuda(MemcpyAsync( 
            g_dptrAOS_PosMass, 
            g_hostAOS_PosMass, 
            4*g_N*sizeof(float), 
            cudaMemcpyHostToDevice ) );
    }

    switch ( algorithm ) {
        case CPU_AOS:
            *ms = ComputeGravitation_AOS( 
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
        case CPU_AOS_tiled:
            *ms = ComputeGravitation_AOS_tiled( 
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
        case CPU_SOA:
            *ms = ComputeGravitation_SOA(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = true;
            break;
#ifdef HAVE_SIMD
        case CPU_SIMD:
            *ms = ComputeGravitation_SIMD(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = true;
            break;
#endif
#ifdef HAVE_SIMD_THREADED
        case CPU_SIMD_threaded:
            *ms = ComputeGravitation_SIMD_threaded(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = true;
            break;
#endif
#ifdef HAVE_SIMD_OPENMP
        case CPU_SIMD_openmp:
            *ms = ComputeGravitation_SIMD_openmp(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = true;
            break;
#endif
#ifndef NO_CUDA
        case GPU_AOS:
            *ms = ComputeGravitation_GPU_AOS( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            cuda(Memcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_AOS_tiled:
            *ms = ComputeGravitation_GPU_AOS_tiled( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            cuda(Memcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_AOS_tiled_const:
            *ms = ComputeGravitation_GPU_AOS_tiled_const( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            cuda(Memcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
#if 0
// commented out - too slow even on SM 3.0
        case GPU_Atomic:
            cuda(Memset( g_dptrAOS_Force, 0, 3*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Atomic( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            cuda(Memcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
#endif
        case GPU_Shared:
            cuda(Memset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Shared( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            cuda(Memcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_Const:
            cuda(Memset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeNBodyGravitation_GPU_AOS_const( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            cuda(Memcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_Shuffle:
            cuda(Memset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Shuffle( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            cuda(Memcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case multiGPU_SingleCPUThread:
            memset( g_hostAOS_Force, 0, 3*g_N*sizeof(float) );
            *ms = ComputeGravitation_multiGPU_singlethread( 
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
        case multiGPU_MultiCPUThread:
            memset( g_hostAOS_Force, 0, 3*g_N*sizeof(float) );
            *ms = ComputeGravitation_multiGPU_threaded( 
                g_hostAOS_Force,
                g_hostAOS_PosMass,
                g_softening*g_softening,
                g_N );
            break;
#endif
        default:
            fprintf(stderr, "Unrecognized algorithm index: %d\n", algorithm);
            abort();
            break;
    }

    if ( g_bGPUCrossCheck ) {
        int cDisagreements = 0;
        for ( int i = 0; i < g_numGPUs; i++ ) {
            for ( int j = 1; j < g_numGPUs; j++ ) {
                if ( memcmp( g_hostAOS_gpuCrossCheckForce[i], 
                             g_hostAOS_gpuCrossCheckForce[j], 
                             3*g_N*sizeof(float) ) ) {
                    fprintf( stderr, "GPU %d and GPU %d disagreed\n", i, j );
                    cDisagreements += 1;
                }
            }
        }
        if ( cDisagreements ) {
            goto Error;
        }
    }


    // SOA -> AOS
    if ( bSOA ) {
        for ( size_t i = 0; i < g_N; i++ ) {
            g_hostAOS_Force[3*i+0] = g_hostSOA_Force[0][i];
            g_hostAOS_Force[3*i+1] = g_hostSOA_Force[1][i]; 
            g_hostAOS_Force[3*i+2] = g_hostSOA_Force[2][i];
        }
    }

    integrateGravitation_AOS( 
        g_hostAOS_PosMass,
        g_hostAOS_VelInvMass,
        g_hostAOS_Force,
        g_dt,
        g_damping,
        g_N );

    if ( g_bGPUCrossCheck && g_fGPUCrosscheckInput ) {
        if ( memcmp( g_hostAOS_Force, g_hostAOS_Force_Golden, 3*g_N*sizeof(float) ) ) {
            printf( "GPU CROSSCHECK FAILURE: Disagreement with golden values\n" );
            goto Error;
        }
    }


    *maxRelError = 0.0f;

    if ( bCrossCheck ) {
        float max = 0.0f;
        for ( size_t i = 0; i < 3*g_N; i++ ) {
            float err = relError( g_hostAOS_Force[i], g_hostAOS_Force_Golden[i] );
            if ( err > max ) {
                max = err;
            }
        }
        *maxRelError = max;
    }
    else {
        KahanAdder sumX;
        KahanAdder sumY;
        KahanAdder sumZ;
        for ( size_t i = 0; i < g_N; i++ ) {
            sumX += g_hostAOS_Force[i*3+0];
            sumY += g_hostAOS_Force[i*3+1];
            sumZ += g_hostAOS_Force[i*3+2];
        }
        *maxRelError = std::max( fabs(sumX), std::max(fabs(sumY), fabs(sumZ)) );
        if ( g_ZeroThreshold != 0.0 && 
             fabs( *maxRelError ) > g_ZeroThreshold ) {
            printf( "Maximum sum of forces > threshold (%E > %E)\n",
                *maxRelError,
                g_ZeroThreshold );
            goto Error;
        }
    }

    return true;
Error:
    return false;
}

workerThread *g_CPUThreadPool;
int g_numCPUCores;

workerThread *g_GPUThreadPool;
int g_numGPUs;

struct gpuInit_struct
{
    int iGPU;

    cudaError_t status;
};

void
initializeGPU( void *_p )
{
    cudaError_t status;

    gpuInit_struct *p = (gpuInit_struct *) _p;
    cuda(SetDevice( p->iGPU ) );
    cuda(SetDeviceFlags( cudaDeviceMapHost ) );
    cuda(Free(0) );
Error:
    p->status = status;    
}

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    // kiloparticles
    int kParticles = 4, kMaxIterations = 0;

    if ( 1 == argc ) {
        printf( "Usage: nbody --numbodies <N> [--nocpu] [--nocrosscheck] [--iterations <N>]\n" );
        printf( "    --numbodies is multiplied by 1024 (default is 4)\n" );
        printf( "    By default, the app checks results against a CPU implementation; \n" );
        printf( "    disable this behavior with --nocrosscheck.\n" );
        printf( "    The CPU implementation may be disabled with --nocpu.\n" );
        printf( "    --nocpu implies --nocrosscheck.\n\n" );
        printf( "    --nosimd uses serial CPU implementation instead of SIMD.\n" );
        printf( "    --iterations specifies a fixed number of iterations to execute\n");
        return 1;
    }

    // for reproducible results for a given N
    srand(7);

    {
        g_numCPUCores = processorCount();
        g_CPUThreadPool = new workerThread[g_numCPUCores];
        for ( size_t i = 0; i < g_numCPUCores; i++ ) {
            if ( ! g_CPUThreadPool[i].initialize( ) ) {
                fprintf( stderr, "Error initializing thread pool\n" );
                return 1;
            }
        }
    }

    status = cudaGetDeviceCount( &g_numGPUs );
    g_bCUDAPresent = (cudaSuccess == status) && (g_numGPUs > 0);
    if ( g_bCUDAPresent ) {
        cudaDeviceProp prop;
        cuda(GetDeviceProperties( &prop, 0 ) );
        g_bSM30Present = prop.major >= 3;
    }
    g_bNoCPU = chCommandLineGetBool( "nocpu", argc, argv );
    if ( g_bNoCPU && ! g_bCUDAPresent ) {
        printf( "--nocpu specified, but no CUDA present...exiting\n" );
        exit(1);
    }

    g_bCrossCheck = ! chCommandLineGetBool( "nocrosscheck", argc, argv );
    if ( g_bNoCPU ) {
        g_bCrossCheck = false;
    }
    if ( g_bCrossCheck && chCommandLineGetBool( "nosse", argc, argv ) ) {
        g_bUseSIMDForCrossCheck = false;
    }

    chCommandLineGet( &kParticles, "numbodies", argc, argv );
    g_N = kParticles*1024;

    chCommandLineGet( &kMaxIterations, "iterations", argc, argv);

    // Round down to the nearest multiple of the CPU count (e.g. if we have
    // a system with a CPU count that isn't a power of two, we need to round)
    g_N -= g_N % g_numCPUCores;

    if ( chCommandLineGetBool( "gpu-crosscheck", argc, argv ) ) {
        g_bGPUCrossCheck = true;
    }
    g_bGPUCrossCheck = chCommandLineGetBool( "gpu-crosscheck", argc, argv );
    {
        char *szFilename;
        if ( chCommandLineGet( &szFilename, "gpu-crosscheck-input-file", argc, argv ) ) {
            if ( ! g_bGPUCrossCheck ) {
                fprintf( stderr, "GPU crosscheck input file requires --gpu-crosscheck\n" );
                goto Error;
            }
            g_fGPUCrosscheckInput = fopen( szFilename, "rb" );
            if ( ! g_fGPUCrosscheckInput ) {
                fprintf( stderr, "Could not open %s for input\n", szFilename );
                goto Error;
            }
            {
                int version;
                if ( 1 != fread( &version, sizeof(int), 1, g_fGPUCrosscheckInput ) ) {
                    fprintf( stderr, "Read of version failed\n" );
                    goto Error;
                }
                if ( version != NBODY_GOLDENFILE_VERSION ) {
                    fprintf( stderr, "File version mismatch - generate new golden files!\n" );
                    goto Error;
                }
            }
            if ( 1 != fread( &g_N, sizeof(int), 1, g_fGPUCrosscheckInput ) ) {
                fprintf( stderr, "Read of particle count failed\n" );
                goto Error;
            }
            if ( 1 != fread( &kMaxIterations, sizeof(int), 1, g_fGPUCrosscheckInput ) ) {
                fprintf( stderr, "Read of iteration count failed\n" );
                goto Error;
            }
            printf( "%d iterations specified in input file\n", kMaxIterations );
        }
        if ( chCommandLineGet( &szFilename, "gpu-crosscheck-output-file", argc, argv  ) ) {
            if ( g_fGPUCrosscheckInput ) {
                fprintf( stderr, "Crosscheck input and output files are mutually exclusive. Please specify only one.\n" );
                goto Error;
            }            
            if ( ! g_bGPUCrossCheck ) {
                fprintf( stderr, "GPU crosscheck output file requires --gpu-crosscheck\n" );
                goto Error;
            }
            g_fGPUCrosscheckOutput = fopen( szFilename, "wb" );
            if ( ! g_fGPUCrosscheckOutput ) {
                fprintf( stderr, "Could not open %s for output\n", szFilename );
                goto Error;
            }
            if ( ! kMaxIterations ) {
                fprintf( stderr, "Must specify --iterations when generating output file for GPU cross check.\n" );
                goto Error;
            }
            {
                int version = NBODY_GOLDENFILE_VERSION;
                if ( 1 != fwrite( &version, sizeof(int), 1, g_fGPUCrosscheckOutput ) ) {
                    fprintf( stderr, "Write of version failed\n" );
                    goto Error;
                }
            }

            if ( 1 != fwrite( &g_N, sizeof(int), 1, g_fGPUCrosscheckOutput ) ) {
                fprintf( stderr, "Write of particle count failed\n" );
                goto Error;
            }
            if ( 1 != fwrite( &kMaxIterations, sizeof(int), 1, g_fGPUCrosscheckOutput ) ) {
                fprintf( stderr, "Write of iteration count failed\n" );
                goto Error;
            }
        }
    }

    chCommandLineGet( &g_ZeroThreshold, "zero", argc, argv );

    if ( g_numGPUs ) {
        // optionally override GPU count from command line
        chCommandLineGet( &g_numGPUs, "numgpus", argc, argv );
        g_GPUThreadPool = new workerThread[g_numGPUs];
        for ( size_t i = 0; i < g_numGPUs; i++ ) {
            if ( ! g_GPUThreadPool[i].initialize( ) ) {
                fprintf( stderr, "Error initializing thread pool\n" );
                return 1;
            }
        }
        for ( int i = 0; i < g_numGPUs; i++ ) {
            gpuInit_struct initGPU = {i};
            g_GPUThreadPool[i].delegateSynchronous( 
                initializeGPU, 
                &initGPU );
            if ( cudaSuccess != initGPU.status ) {
                fprintf( stderr, "Initializing GPU %d failed "
                    " with %d (%s)\n",
                    i, 
                    initGPU.status, 
                    cudaGetErrorString( initGPU.status ) );
                return 1;
            }
        }
    }

    printf( "Running simulation with %d particles, crosscheck %s, CPU %s\n", (int) g_N,
        g_bCrossCheck ? "enabled" : "disabled",
        g_bNoCPU ? "disabled" : "enabled" );

#if defined(HAVE_SIMD_OPENMP)
    g_maxAlgorithm = CPU_SIMD_openmp;
#elif defined(HAVE_SIMD_THREADED)
    g_maxAlgorithm = CPU_SIMD_threaded;
#elif defined(HAVE_SIMD)
    g_maxAlgorithm = CPU_SIMD;
#else
    g_maxAlgorithm = CPU_SOA;
#endif
    g_Algorithm = g_bCUDAPresent ? GPU_AOS : g_maxAlgorithm;
	g_Algorithm = multiGPU_SingleCPUThread;
    if ( g_bCUDAPresent || g_bNoCPU ) {
        // max algorithm is different depending on whether SM 3.0 is present
        g_maxAlgorithm = g_bSM30Present ? GPU_AOS_tiled_const : multiGPU_MultiCPUThread;
    }

    if ( g_bCUDAPresent ) {
        cudaDeviceProp propForVersion;

        cuda(SetDeviceFlags( cudaDeviceMapHost ) );
        cuda(GetDeviceProperties( &propForVersion, 0 ) );
        if ( propForVersion.major < 3 ) {
            // Only SM 3.x supports shuffle and fast atomics, so we cannot run
            // some algorithms on this board.
            g_maxAlgorithm = multiGPU_MultiCPUThread;
        }

        cuda(HostAlloc( (void **) &g_hostAOS_PosMass, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        for ( int i = 0; i < 3; i++ ) {
            cuda(HostAlloc( (void **) &g_hostSOA_Pos[i], g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
            cuda(HostAlloc( (void **) &g_hostSOA_Force[i], g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        }
        cuda(HostAlloc( (void **) &g_hostAOS_Force, 3*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        cuda(HostAlloc( (void **) &g_hostAOS_Force_Golden, 3*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        cuda(HostAlloc( (void **) &g_hostAOS_VelInvMass, 4*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        cuda(HostAlloc( (void **) &g_hostSOA_Mass, g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
        cuda(HostAlloc( (void **) &g_hostSOA_InvMass, g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );

        cuda(Malloc( &g_dptrAOS_PosMass, 4*g_N*sizeof(float) ) );
        cuda(Malloc( (void **) &g_dptrAOS_Force, 3*g_N*sizeof(float) ) );

        if ( g_bGPUCrossCheck  ) {
            printf( "GPU cross check enabled (%d GPUs), disabling CPU\n", g_numGPUs );
            g_bNoCPU = true;
            g_bCrossCheck = false;
            if ( g_numGPUs < 2 ) {
                fprintf( stderr, "GPU cross check enabled, but <2 GPUs available\n" );
                goto Error;
            }
            for ( int i = 0; i < g_numGPUs; i++ ) {
                cuda(HostAlloc( (void **) (&g_hostAOS_gpuCrossCheckForce[i]), 3*g_N*sizeof(float), cudaHostAllocPortable|cudaHostAllocMapped ) );
            }
        }
    }
    else {
        g_hostAOS_PosMass = new float[4*g_N];
        for ( int i = 0; i < 3; i++ ) {
            g_hostSOA_Pos[i] = new float[g_N];
            g_hostSOA_Force[i] = new float[g_N];
        }
        g_hostSOA_Mass = new float[g_N];
        g_hostAOS_Force = new float[3*g_N];
        g_hostAOS_Force_Golden = new float[3*g_N];
        g_hostAOS_VelInvMass = new float[4*g_N];
        g_hostSOA_Mass = new float[g_N];
        g_hostSOA_InvMass = new float[g_N];
    }

    randomUnitBodies( g_hostAOS_PosMass, g_hostAOS_VelInvMass, g_N );
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostSOA_Mass[i] = g_hostAOS_PosMass[4*i+3];
        g_hostSOA_InvMass[i] = 1.0f / g_hostSOA_Mass[i];
    }

#if 0
    // gather performance data over GPU implementations
    // for different problem sizes.

    printf( "kBodies\t" );
    for ( int algorithm = GPU_AOS; 
              algorithm < sizeof(rgszAlgorithmNames)/sizeof(rgszAlgorithmNames[0]); 
              algorithm++ ) {
        printf( "%s\t", rgszAlgorithmNames[algorithm] );
    }
    printf( "\n" );

    for ( int kBodies = 3; kBodies <= 96; kBodies += 3 ) {

	g_N = 1024*kBodies;

        printf( "%d\t", kBodies );

	for ( int algorithm = GPU_AOS; 
                  algorithm < sizeof(rgszAlgorithmNames)/sizeof(rgszAlgorithmNames[0]); 
                  algorithm++ ) {
            float sum = 0.0f;
            const int numIterations = 10;
            for ( int i = 0; i < numIterations; i++ ) {
                float ms, err;
		if ( ! ComputeGravitation( &ms, &err, (nbodyAlgorithm_enum) algorithm, g_bCrossCheck ) ) {
			fprintf( stderr, "Error computing timestep\n" );
			exit(1);
		}
                sum += ms;
            }
            sum /= (float) numIterations;

            double interactionsPerSecond = (double) g_N*g_N*1000.0f / sum;
            if ( interactionsPerSecond > 1e9 ) {
                printf ( "%.2f\t", interactionsPerSecond/1e9 );
            }
            else {
                printf ( "%.3f\t", interactionsPerSecond/1e9 );               
            }
        }
        printf( "\n" );
    }
    return 0;
#endif
    {
        int kIterations = 0;
        bool bStop = false;
        while ( ! bStop ) {
            float ms, err;

            if ( ! ComputeGravitation( &ms, &err, g_Algorithm, g_bCrossCheck ) ) {
                fprintf( stderr, "Error computing timestep\n" );
                exit(1);
            }
            double interactionsPerSecond = (double) g_N*g_N*1000.0f / ms;
            if ( interactionsPerSecond > 1e9 ) {
                printf ( "\r%s: %8.2f ms = %8.3fx10^9 interactions/s (Rel. error: %E)\n",
                    rgszAlgorithmNames[g_Algorithm], 
                    ms, 
                    interactionsPerSecond/1e9, 
                    err );
            }
            else {
                printf ( "\r%s: %8.2f ms = %8.3fx10^6 interactions/s (Rel. error: %E)\n",
                    rgszAlgorithmNames[g_Algorithm], 
                    ms, 
                    interactionsPerSecond/1e6, 
                    err );
            }
            if (kMaxIterations) {
                kIterations++;
                if (kIterations >= kMaxIterations) {
                    bStop = true;
                }
            }
            if ( kbhit() ) {
                char c = getch();
                switch ( c ) {
                    case ' ':
                        if ( g_Algorithm == g_maxAlgorithm ) {
                            g_Algorithm = g_bNoCPU ? GPU_AOS : CPU_AOS;
                            // Skip slow CPU implementations if we are using SIMD for cross-check
                            if ( g_bUseSIMDForCrossCheck ) {
#if defined(HAVE_SIMD_THREADED)
                                g_Algorithm = CPU_SIMD_threaded;
#elif defined(HAVE_SIMD_OPENMP)
                                g_Algorithm = CPU_SIMD_openmp;
#endif
                            }
                        }
                        else {
                            g_Algorithm = (enum nbodyAlgorithm_enum) (g_Algorithm+1);
                        }
                        break;
                    case 'q':
                    case 'Q':
                        bStop = true;
                        break;
                }

            }
        }
    }

    if ( g_fGPUCrosscheckInput ) fclose( g_fGPUCrosscheckInput );
    if ( g_fGPUCrosscheckOutput ) fclose( g_fGPUCrosscheckOutput );

    return 0;
Error:
    if ( g_fGPUCrosscheckInput ) fclose( g_fGPUCrosscheckInput );
    if ( g_fGPUCrosscheckOutput ) fclose( g_fGPUCrosscheckOutput );
    if ( cudaSuccess != status ) {
        printf( "CUDA Error: %s\n", cudaGetErrorString( status ) );
    }
    return 1;
}
