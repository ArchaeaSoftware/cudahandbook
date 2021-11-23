/*
 *
 * nbody_GPU_Shuffle.h
 *
 * Warp shuffle-based implementation of the O(N^2) N-body calculation.
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
__global__ void
ComputeNBodyGravitation_Shuffle( 
    float *force, 
    float *posMass, 
    float softeningSquared, 
    size_t N )
{
    const int laneid = threadIdx.x & 31;
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x )
    {
        float acc[3] = {0};
        float4 myPosMass = ((float4 *) posMass)[i];

        for ( int j = 0; j < N; j += 32 ) {
            float4 shufSrcPosMass = ((float4 *) posMass)[j+laneid];
#pragma unroll 32
            for ( int k = 0; k < 32; k++ ) {
                float fx, fy, fz;
                float4 shufDstPosMass;

                shufDstPosMass.x = __shfl_sync( 0xffffffff, shufSrcPosMass.x, k );
                shufDstPosMass.y = __shfl_sync( 0xffffffff, shufSrcPosMass.y, k );
                shufDstPosMass.z = __shfl_sync( 0xffffffff, shufSrcPosMass.z, k );
                shufDstPosMass.w = __shfl_sync( 0xffffffff, shufSrcPosMass.w, k );

                bodyBodyInteraction(
                    &fx, &fy, &fz, 
                    myPosMass.x, myPosMass.y, myPosMass.z, 
                    shufDstPosMass.x, 
                    shufDstPosMass.y, 
                    shufDstPosMass.z, 
                    shufDstPosMass.w, 
                    softeningSquared);
                acc[0] += fx;
                acc[1] += fy;
                acc[2] += fz;
            }
        }

        force[3*i+0] = acc[0];
        force[3*i+1] = acc[1];
        force[3*i+2] = acc[2];
    }
}
#else
//
// If SM 3.x not available, use naive algorithm
//
__global__ void
ComputeNBodyGravitation_Shuffle( float *force, float *posMass, float softeningSquared, size_t N )
{
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x )
    {
        float acc[3] = {0};
        float4 me = ((float4 *) posMass)[i];
        float myX = me.x;
        float myY = me.y;
        float myZ = me.z;
        for ( int j = 0; j < N; j++ ) {
            float fx, fy, fz;
            float4 body = ((float4 *) posMass)[j];
            bodyBodyInteraction( &fx, &fy, &fz, myX, myY, myZ, body.x, body.y, body.z, body.w, softeningSquared);
            acc[0] += fx;
            acc[1] += fy;
            acc[2] += fz;
        }
        force[3*i+0] = acc[0];
        force[3*i+1] = acc[1];
        force[3*i+2] = acc[2];
    }
}
#endif

float
ComputeGravitation_GPU_Shuffle( float *force, float *posMass, float softeningSquared, size_t N )
{
    cudaError_t status;
    cudaEvent_t evStart = 0, evStop = 0;
    float ms = 0.0f;
    cuda(EventCreate( &evStart ) );
    cuda(EventCreate( &evStop ) );
    cuda(EventRecord( evStart, NULL ) );
    ComputeNBodyGravitation_Shuffle <<<300,256>>>( force, posMass, softeningSquared, N );
    cuda(EventRecord( evStop, NULL ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( &ms, evStart, evStop ) );
Error:
    cuda(EventDestroy( evStop ) );
    cuda(EventDestroy( evStart ) );
    return ms;
}
