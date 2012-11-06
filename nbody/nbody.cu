/*
 *
 * nbody.cu
 *
 * N-body example that illustrates gravitational simulation.
 * This is the type of computation that GPUs excel at:
 * parallelizable, with lots of FLOPS per unit of external 
 * memory bandwidth required.
 *
 * Build with: nvcc -I ../chLib <options> nbody.cu
 *   On Linux: nvcc -I ../chLib <options> nbody.cu -lpthread -lrt
 * Requires: No minimum SM requirement.
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

float *g_hostAOS_PosMass;
float *g_hostAOS_VelInvMass;
float *g_hostAOS_Force;

float *g_dptrAOS_PosMass;
float *g_dptrAOS_Force;


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
#include "nbody_CPU_SOA.h"
#include "nbody_CPU_SSE.h"
#include "nbody_CPU_SSE_threaded.h"

#include "nbody_GPU_AOS.cuh"
#include "nbody_GPU_Shared.cuh"
#include "nbody_GPU_Shuffle.cuh"
#include "nbody_GPU_Atomic.cuh"

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

enum nbodyAlgorithm_enum {
    CPU_AOS = 0,    /* This is the golden implementation */
    CPU_SOA,
    CPU_SSE,
    CPU_SSE_threaded,
    GPU_AOS,
    GPU_Atomic,
    GPU_Shared,
    GPU_Shuffle/*,
    GPU_SOA,
    MultiGPU*/
};

const char *rgszAlgorithmNames[] = { "CPU_AOS", "CPU_SOA", "CPU_SSE", "CPU_SSE_threaded", "GPU_AOS", "GPU_Atomic", "GPU_Shared", "GPU_Shuffle" };

enum nbodyAlgorithm_enum g_Algorithm = CPU_SSE_threaded;

//
// g_maxAlgorithm is used to determine when to rotate g_Algorithm back to CPU_AOS
// If CUDA is present, it is CPU_SSE_threaded, otherwise GPU_Shuffle
// The CPU and GPU algorithms must be contiguous, and the logic in main() to
// initialize this value must be modified if any new algorithms are added.
//
enum nbodyAlgorithm_enum g_maxAlgorithm;
bool g_bCrossCheck = true;

bool
ComputeGravitation( 
    float *ms,
    float *maxRelError,
    nbodyAlgorithm_enum algorithm, 
    bool bCrossCheck )
{
    cudaError_t status;
    bool bSOA = false;

    if ( bCrossCheck ) {
        ComputeGravitation_AOS( 
            g_hostAOS_Force_Golden,
            g_hostAOS_PosMass,
            g_softening*g_softening,
            g_N );
    }

    // AOS -> SOA data structures in case we are measuring SOA performance
    for ( size_t i = 0; i < g_N; i++ ) {
        g_hostSOA_Pos[0][i]  = g_hostAOS_PosMass[4*i+0];
        g_hostSOA_Pos[1][i]  = g_hostAOS_PosMass[4*i+1];
        g_hostSOA_Pos[2][i]  = g_hostAOS_PosMass[4*i+2];
        g_hostSOA_Mass[i]    = g_hostAOS_PosMass[4*i+3];
        g_hostSOA_InvMass[i] = 1.0f / g_hostSOA_Mass[i];
    }

    // CPU->GPU copies in case we are measuring GPU performance
    if ( g_bCUDAPresent ) {
        CUDART_CHECK( cudaMemcpyAsync( g_dptrAOS_PosMass, g_hostAOS_PosMass, 4*g_N*sizeof(float), cudaMemcpyHostToDevice ) );
    }

    switch ( algorithm ) {
        case CPU_AOS:
            *ms = ComputeGravitation_AOS( 
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
        case CPU_SSE:
            *ms = ComputeGravitation_SSE(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = true;
            break;
        case CPU_SSE_threaded:
            *ms = ComputeGravitation_SSE_threaded(
                g_hostSOA_Force,
                g_hostSOA_Pos,
                g_hostSOA_Mass,
                g_softening*g_softening,
                g_N );
            bSOA = true;
            break;
        case GPU_AOS:
            *ms = ComputeGravitation_GPU_AOS( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_Atomic:
            CUDART_CHECK( cudaMemset( g_dptrAOS_Force, 0, 3*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Atomic( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_Shared:
            CUDART_CHECK( cudaMemset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Shared( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
        case GPU_Shuffle:
            CUDART_CHECK( cudaMemset( g_dptrAOS_Force, 0, 3*g_N*sizeof(float) ) );
            *ms = ComputeGravitation_GPU_Shuffle( 
                g_dptrAOS_Force,
                g_dptrAOS_PosMass,
                g_softening*g_softening,
                g_N );
            CUDART_CHECK( cudaMemcpy( g_hostAOS_Force, g_dptrAOS_Force, 3*g_N*sizeof(float), cudaMemcpyDeviceToHost ) );
            break;
    }

    // SOA -> AOS
    if ( bSOA ) {
        for ( size_t i = 0; i < g_N; i++ ) {
            g_hostAOS_Force[3*i+0] = g_hostSOA_Force[0][i];
            g_hostAOS_Force[3*i+1] = g_hostSOA_Force[1][i];
            g_hostAOS_Force[3*i+2] = g_hostSOA_Force[2][i];
        }
    }

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

    integrateGravitation_AOS( 
        g_hostAOS_PosMass,
        g_hostAOS_VelInvMass,
        g_hostAOS_Force,
        g_dt,
        g_damping,
        g_N );
    return true;
Error:
    return false;
}

workerThread *g_ThreadPool;
size_t g_cThreads;

int
main( int argc, char *argv[] )
{
    cudaError_t status;
    // kiloparticles
    int kParticles = 4;

    {
        g_cThreads = processorCount()/2;
        g_ThreadPool = new workerThread[g_cThreads];
        for ( size_t i = 0; i < g_cThreads; i++ ) {
            if ( ! g_ThreadPool[i].initialize( ) ) {
                fprintf( stderr, "Error initializing thread pool\n" );
                return 1;
            }
        }
/*
for ( int i = 0; i < g_cThreads; i++ ) {
            if ( ! g_ThreadPool[i].initialize() ) {
                fprintf( stderr, "Error creating thread pool\n" );
            }
        }*/

    }

    chCommandLineGet( &kParticles, "numbodies", argc, argv );
    g_N = kParticles*1024;
    printf( "Running simulation with %d particles\n", (int) g_N );

    status = cudaSetDeviceFlags( cudaDeviceMapHost );
    g_bCUDAPresent = (cudaSuccess == status);
    g_maxAlgorithm = g_bCUDAPresent ? GPU_Shuffle : CPU_SSE_threaded;

    if ( g_bCUDAPresent ) {

        CUDART_CHECK( cudaSetDeviceFlags( cudaDeviceMapHost ) );

        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_PosMass, 4*g_N*sizeof(float), cudaHostAllocPortable ) );
        for ( int i = 0; i < 3; i++ ) {
            CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Pos[i], g_N*sizeof(float), cudaHostAllocPortable ) );
            CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Force[i], g_N*sizeof(float), cudaHostAllocPortable ) );
        }
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_Force, 3*g_N*sizeof(float), cudaHostAllocPortable ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_Force_Golden, 3*g_N*sizeof(float), cudaHostAllocPortable ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostAOS_VelInvMass, 4*g_N*sizeof(float), cudaHostAllocPortable ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_Mass, g_N*sizeof(float), cudaHostAllocPortable ) );
        CUDART_CHECK( cudaHostAlloc( (void **) &g_hostSOA_InvMass, g_N*sizeof(float), cudaHostAllocPortable ) );

        CUDART_CHECK( cudaMalloc( &g_dptrAOS_PosMass, 4*g_N*sizeof(float) ) );
        CUDART_CHECK( cudaMalloc( (void **) &g_dptrAOS_Force, 3*g_N*sizeof(float) ) );
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

    {
        bool bStop = false;
        while ( ! bStop ) {
            float ms, err;

            ComputeGravitation( &ms, &err, g_Algorithm, g_bCrossCheck );
            double interactionsPerSecond = (double) g_N*g_N*1000.0f / ms;
            if ( interactionsPerSecond > 1e9 ) {
                printf ( "%s: %.2f ms = %.3fx10^9 interactions/s (Rel. error: %E)\n", 
                    rgszAlgorithmNames[g_Algorithm], 
                    ms, 
                    interactionsPerSecond/1e9, 
                    err );
            }
            else {
                printf ( "%s: %.2f ms = %.3fx10^6 interactions/s (Rel. error: %E)\n", 
                    rgszAlgorithmNames[g_Algorithm], 
                    ms, 
                    interactionsPerSecond/1e6, 
                    err );
            }
            if ( kbhit() ) {
                char c = getch();
                switch ( c ) {
                    case ' ':
                        if ( g_Algorithm == g_maxAlgorithm ) {
                            g_Algorithm = CPU_AOS;
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

    return 0;
Error:
    if ( cudaSuccess != status ) {
        printf( "CUDA Error: %s\n", cudaGetErrorString( status ) );
    }
    return 1;
}
