/*
 *
 * nbody_GPU_AOS_tiled.cuh
 *
 * CUDA implementation of the O(N^2) N-body calculation.
 * Tiled to take advantage of the symmetry of gravitational
 * forces: Fij=-Fji
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

template<int nTile, typename T>
__device__ void
DoDiagonalTile_GPU( 
    float *force, 
    float *posMass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    size_t i = iTile*nTile+threadIdx.x;
    float acc[3] = {0, 0, 0};
    float myX = posMass[i*4+0];
    float myY = posMass[i*4+1];
    float myZ = posMass[i*4+2];

if ( 0 == threadIdx.x ) printf( "Computing tile %d (%d..%d)\n", i, i+blockDim.x);

    for ( size_t _j = 0; _j < nTile; _j++ ) {
        size_t j = jTile*nTile+_j;

        float fx, fy, fz;
        float bodyX = posMass[j*4+0];
        float bodyY = posMass[j*4+1];
        float bodyZ = posMass[j*4+2];
        float bodyMass = posMass[j*4+3];

        bodyBodyInteraction<float>(
            &fx, &fy, &fz,
            myX, myY, myZ,
            bodyX, bodyY, bodyZ, bodyMass,
            softeningSquared );
        acc[0] += fx;
        acc[1] += fy;
        acc[2] += fz;
    }

    atomicAdd( &force[3*i+0], acc[0] );
    atomicAdd( &force[3*i+1], acc[1] );
    atomicAdd( &force[3*i+2], acc[2] );
}



template<int nTile, typename T>
__global__ void
ComputeNBodyGravitation_GPU_tiled( T *force, T *posMass, size_t N, T softeningSquared )
{
    for ( int iTile = blockIdx.x*blockDim.x;
              iTile < N/nTile;
              iTile += blockDim.x*gridDim.x )
    {
        for ( int jTile = blockIdx.x*blockDim.x;
                  jTile < N/nTile;
                  jTile += blockDim.x*gridDim.x )
        {
            DoDiagonalTile_GPU<32,T>( force, posMass, softeningSquared, iTile, jTile );
        }
    }
}

float
ComputeGravitation_GPU_AOS_tiled(
    float *force, 
    float *posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;
    cudaEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;
    cudaPrintfInit();
    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );
    CUDART_CHECK( cudaEventRecord( evStart, NULL ) );
    CUDART_CHECK( cudaMemset( force, 0, 3*N*sizeof(float) ) );
    ComputeNBodyGravitation_GPU_tiled<32,float> <<<300,32>>>( force, posMass, N, softeningSquared );
    CUDART_CHECK( cudaEventRecord( evStop, NULL ) );
    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( &ms, evStart, evStop ) );
Error:
    CUDART_CHECK( cudaEventDestroy( evStop ) );
    CUDART_CHECK( cudaEventDestroy( evStart ) );
    cudaPrintfDisplay();
    return ms;
}

