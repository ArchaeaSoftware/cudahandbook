/*
 *
 * nbody.h
 *
 * Header file to declare globals in nbody.cu
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

#ifndef __CUDAHANDBOOK_NBODY_H__
#define __CUDAHANDBOOK_NBODY_H__

//#include "nbody_CPU_SIMD.h"

#include <chThread.h>
//#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

extern bool g_bCUDAPresent;
extern bool g_bGPUCrossCheck;

extern bool g_GPUCrosscheck;
#define NBODY_GOLDENFILE_VERSION 0x100
extern FILE *g_fGPUCrosscheckInput;
extern FILE *g_fGPUCrosscheckOutput;

// for GPU cross-check
const int g_maxGPUs = 32;
extern float *g_hostAOS_gpuCrossCheckForce[g_maxGPUs];

extern float *g_dptrAOS_PosMass;
extern float *g_dptrAOS_Force;


extern float *g_hostSOA_Pos[3];
extern float *g_hostSOA_Force[3];
extern float *g_hostSOA_Mass;
extern float *g_hostSOA_InvMass;

extern size_t g_N;

extern float g_softening;
extern float g_damping;
extern float g_dt;

template<typename T>
struct PosMass {
    T x_, y_, z_, mass_;
};

template<typename T>
struct VelInvMass {
    T dx_, dy_, dz_, invMass_;
};

template<typename T>
struct Force3D {
    T ddx_, ddy_, ddz_;
};

// base implementation has AOS on CPU
// GPU implementations inherit from this class because they need 
// system memory copies of everything anyway
template<typename T>
class NBodyAlgorithm {
public:
    inline NBodyAlgorithm<T>() { }
    virtual ~NBodyAlgorithm<T>() { }
    virtual bool Initialize( size_t N, int seed, T softening );
    virtual const char *getAlgoName() const { return "CPU AOS"; }

    size_t N() const { return N_; }
    T softening() const { return softening_; }

    // return value is elapsed time needed for the time step
    virtual float computeTimeStep( );//std::vector<Force3D<T> >& force );
    virtual void integrateGravitation( T dt, T damping );

    // accessors
    void setBody( size_t i, const PosMass<T>& body ) { posMass_[i] = body; }
    PosMass<T>& getBody( size_t i ) { return posMass_[i]; }
    PosMass<T> getBody( size_t i ) const { return posMass_[i]; }

    const std::vector<Force3D<T>>& force() const { return force_; }
    const std::vector<PosMass<T>>& posMass() const { return posMass_; }
    const std::vector<VelInvMass<T>>& velInvMass() const { return velInvMass_; }

    std::vector<Force3D<T>>& force() { return force_; }
    std::vector<PosMass<T>>& posMass() { return posMass_; }
    std::vector<VelInvMass<T>>& velInvMass() { return velInvMass_; }

private:
    size_t N_;

    T softening_;

    std::vector<Force3D<T>> force_;
    std::vector<PosMass<T>> posMass_;
    std::vector<VelInvMass<T>> velInvMass_;
};

template<typename T>
class NBodyAlgorithm_GPU : public NBodyAlgorithm<T> {
public:
    inline NBodyAlgorithm_GPU<T>() { evStart_ = evStop_ = nullptr; }
    virtual ~NBodyAlgorithm_GPU<T>() { 
        cudaEventDestroy( evStart_ );
        cudaEventDestroy( evStop_ );
    }
    virtual const char *getAlgoName() const { return "GPU AOS"; }
    virtual bool Initialize( size_t N, int seed, T softening );
    virtual float computeTimeStep( );

    // Processes i'th subarray for the timestep.
    // This virtual function is used to explore different GPU
    // implementations without duplicating the multi-GPU code.
    //virtual float gpuComputeTimeSubstep( size_t i );
private:
    cudaEvent_t evStart_, evStop_;

    std::vector<thrust::device_vector<Force3D<float>>> gpuForce_;
    std::vector<thrust::device_vector<PosMass<float>>> gpuPosMass_;
    std::vector<thrust::device_vector<VelInvMass<float>>> gpuVelInvMass_;
};

struct alignas(32) aligned_float {
    float f_;
    aligned_float( float f) { f_ = f; }
    operator float() { return f_; }
};

struct alignas(32) aligned_double {
    float d_;
    aligned_double( double d) { d_ = d; }
    operator double() { return d_; }
};

template<typename T>
class NBodyAlgorithm_SOA : public NBodyAlgorithm<T> {
public:
    NBodyAlgorithm_SOA<T>() { }
    virtual ~NBodyAlgorithm_SOA<T>() { }
    virtual const char *getAlgoName() const { return "CPU SOA"; }
    virtual bool Initialize( size_t N, int seed, T softening );
    virtual float computeTimeStep( );

    std::vector<T>& x() { return x_; }
    std::vector<T>& y() { return y_; }
    std::vector<T>& z() { return z_; }
    std::vector<T>& mass() { return mass_; }
    std::vector<T>& ddx() { return ddx_; }
    std::vector<T>& ddy() { return ddy_; }
    std::vector<T>& ddz() { return ddz_; }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<T>& mass() const { return mass_; }
    const std::vector<T>& ddx() const { return ddx_; }
    const std::vector<T>& ddy() const { return ddy_; }
    const std::vector<T>& ddz() const { return ddz_; }

private:
    std::vector<T> x_, y_, z_, mass_; // use aligned_float for 32B alignment
    std::vector<T> ddx_, ddy_, ddz_;  // force
};

template<typename T>
class NBodyAlgorithm_SSE : public NBodyAlgorithm_SOA<T> {
public:
    NBodyAlgorithm_SSE<T>() { }
    virtual ~NBodyAlgorithm_SSE<T>() { }
    virtual const char *getAlgoName() const { return "CPU SSE"; }
    virtual float computeTimeStep( );
};

template<typename T>
class NBodyAlgorithm_AVX : public NBodyAlgorithm_SSE<T> {
public:
    NBodyAlgorithm_AVX<T>() { }
    virtual ~NBodyAlgorithm_AVX<T>() { }
    virtual const char *getAlgoName() const { return "CPU AVX"; }
    virtual float computeTimeStep( );
};

template<typename T>
class NBodyAlgorithm_FMA : public NBodyAlgorithm_SSE<T> {
public:
    NBodyAlgorithm_FMA<T>() { }
    virtual ~NBodyAlgorithm_FMA<T>() { }
    virtual const char *getAlgoName() const { return "CPU FMA"; }
    virtual float computeTimeStep( );
};

enum nbodyAlgorithm_enum {
    CPU_AOS = 0,    /* This is the golden implementation */
    CPU_AOS_tiled,
    CPU_SOA,
#ifdef HAVE_SIMD
    CPU_SIMD,
#endif
#ifdef HAVE_SIMD_THREADED
    CPU_SIMD_threaded,
#endif
#ifdef HAVE_SIMD_OPENMP
    CPU_SIMD_openmp,
#endif
    GPU_AOS,
    GPU_Shared,
    GPU_Const,
    multiGPU_SingleCPUThread,
    multiGPU_MultiCPUThread,
// SM 3.0 only
    GPU_Shuffle,
    GPU_AOS_tiled,
    GPU_AOS_tiled_const,
//    GPU_Atomic
};


static const char *rgszAlgorithmNames[] = { 
    "CPU_AOS", 
    "CPU_AOS_tiled", 
    "CPU_SOA", 
#ifdef HAVE_SIMD
    "CPU_SIMD",
#endif
#ifdef HAVE_SIMD_THREADED
    "CPU_SIMD_threaded",
#endif
#ifdef HAVE_SIMD_OPENMP
    "CPU_SIMD_openmp",
#endif
    "GPU_AOS", 
    "GPU_Shared", 
    "GPU_Const",
    "multiGPU_SingleCPUThread",
    "multiGPU_MultiCPUThread",
// SM 3.0 only
    "GPU_Shuffle",
    "GPU_AOS_tiled",
    "GPU_AOS_tiled_const",
//    "GPU_Atomic"
};

extern const char *rgszAlgorithmNames[];

extern enum nbodyAlgorithm_enum g_Algorithm;

//
// g_maxAlgorithm is used to determine when to rotate g_Algorithm back to CPU_AOS
// If CUDA is present, it is CPU_SIMD_threaded, otherwise GPU_Shuffle
// The CPU and GPU algorithms must be contiguous, and the logic in main() to
// initialize this value must be modified if any new algorithms are added.
//
extern enum nbodyAlgorithm_enum g_maxAlgorithm;
extern bool g_bCrossCheck;
extern bool g_bNoCPU;

extern cudahandbook::threading::workerThread *g_CPUThreadPool;
extern int g_numCPUCores;

extern int g_numGPUs;
extern cudahandbook::threading::workerThread *g_GPUThreadPool;

//extern float *g_hostAOS_PosMass;
extern std::vector<PosMass<float>> g_hostAOS_PosMass;
extern std::vector<VelInvMass<float>> g_hostAOS_VelInvMass;
extern std::vector<Force3D<float>> g_hostAOS_Force;

// Buffer to hold the golden version of the forces, used for comparison
// Along with timing results, we report the maximum relative error with 
// respect to this array.
extern std::vector<Force3D<float>> g_hostAOS_Force_Golden;



#if 0
extern float ComputeGravitation_GPU_Shared           ( float *force, float *posMass, float softeningSquared, size_t N );
extern float ComputeGravitation_multiGPU_singlethread( float *force, float *posMass, float softeningSquared, size_t N );
extern float ComputeGravitation_multiGPU_threaded    ( float *force, float *posMass, float softeningSquared, size_t N );
#endif


#endif
