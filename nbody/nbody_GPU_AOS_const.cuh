/*
 *
 * nbody_GPU_AOS_const.h
 *
 * CUDA implementation of the O(N^2) N-body calculation.
 * Uses __constant__ memory to hold one set of input body
 * descriptions - requires multiple asynchronous memcpys
 * and kernel launches to cycle through the calculation.
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

const int g_bodiesPerPass = 4000;
__constant__ __device__ float4 g_constantBodies[g_bodiesPerPass];

template<typename T>
__global__ void
ComputeNBodyGravitation_GPU_AOS_const( 
    T *force, 
    T *posMass, 
    T softeningSquared,
    size_t n, 
    size_t N )
{
    for ( int i = blockIdx.x*blockDim.x + threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x )
    {
        T acc[3] = {0};
        float4 me = ((float4 *) posMass)[i];
        T myX = me.x;
        T myY = me.y;
        T myZ = me.z;
        for ( int j = 0; j < n; j++ ) {
            float4 body = g_constantBodies[j];
            float fx, fy, fz;
            bodyBodyInteraction( 
                &fx, &fy, &fz, 
                myX, myY, myZ, 
                body.x, body.y, body.z, body.w, 
                softeningSquared);
            acc[0] += fx;
            acc[1] += fy;
            acc[2] += fz;
        }
        force[3*i+0] += acc[0];
        force[3*i+1] += acc[1];
        force[3*i+2] += acc[2];
    }
}

float
ComputeNBodyGravitation_GPU_AOS_const(
    float *force, 
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;
    cudaEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;
    size_t bodiesLeft = N;

    void *p;
    cuda(GetSymbolAddress( &p, g_constantBodies ) );

    cuda(EventCreate( &evStart ) );
    cuda(EventCreate( &evStop ) );
    cuda(EventRecord( evStart, NULL ) );
    for ( size_t i = 0; i < N; i += g_bodiesPerPass ) {
        // bodiesThisPass = max(bodiesLeft, g_bodiesPerPass);
        size_t bodiesThisPass = bodiesLeft;
        if ( bodiesThisPass > g_bodiesPerPass ) {
            bodiesThisPass = g_bodiesPerPass;
        }
        cuda(MemcpyToSymbolAsync( 
            g_constantBodies,
            ((float4 *) posMass)+i,
            bodiesThisPass*sizeof(float4),
            0,
            cudaMemcpyDeviceToDevice,
            NULL ) );
        ComputeNBodyGravitation_GPU_AOS_const<float> <<<300,256>>>( 
            force, posMass, softeningSquared, bodiesThisPass, N );
        bodiesLeft -= bodiesThisPass;
    }
    cuda(EventRecord( evStop, NULL ) );
    cuda(DeviceSynchronize() );
    cuda(EventElapsedTime( &ms, evStart, evStop ) );
Error:
    cudaEventDestroy( evStop );
    cudaEventDestroy( evStart );
    return ms;
}
