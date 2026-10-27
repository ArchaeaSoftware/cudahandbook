/*
 *
 * nbody_multiGPU_tile.cuh
 *
 * Shared memory-based implementation of the O(N^2) N-body calculation.
 * This header is designed to be included by the multi-GPU versions.
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
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

#ifndef __CUDAHANDBOOK_NBODY_MULTIGPU_TILE_CUH__
#define __CUDAHANDBOOK_NBODY_MULTIGPU_TILE_CUH__

#include "bodybodyInteraction.cuh"

//
// Computes the forces that one block of source bodies exerts on one block of
// target bodies, and adds them to whatever the force array already holds.
//
// Taking the source block as a parameter, rather than looping over all N
// bodies, is what lets a decomposition hand a device one tile of the force
// matrix at a time: the ring calls this once per block as that block arrives,
// and the two-dimensional decomposition calls it once for the single tile
// that device owns.  Accumulating rather than overwriting is what lets those
// partial results be summed, whether by successive calls on one device or by
// a collective across several.
//
// The target and source blocks are both indexed into the same position array,
// so a device that holds the whole array can be handed any tile of the
// matrix without rearranging its memory.
//
inline __device__ void
ComputeNBodyGravitation_Tile( 
    float *force, 
    float *posMass, 
    float softeningSquared, 
    size_t base,        // index of the first target body
    size_t n,           // number of target bodies
    size_t srcBase,     // index of the first source body
    size_t srcN )       // number of source bodies
{
    extern __shared__ float4 shPosMass[];
    for ( int m = blockIdx.x*blockDim.x + threadIdx.x;
              m < n;
              m += blockDim.x*gridDim.x )
    {
        size_t i = base+m;
        float acc[3] = {0};
        float4 myPosMass = ((float4 *) posMass)[i];
        for ( size_t j = 0; j < srcN; j += blockDim.x ) {
            shPosMass[threadIdx.x] = ((float4 *) posMass)[srcBase+j+threadIdx.x];
            __syncthreads();
            for ( size_t k = 0; k < blockDim.x; k++ ) {
                float fx, fy, fz;
                float4 bodyPosMass = shPosMass[k];

                bodyBodyInteraction( 
                    &fx, &fy, &fz, 
                    myPosMass.x, myPosMass.y, myPosMass.z, 
                    bodyPosMass.x, 
                    bodyPosMass.y, 
                    bodyPosMass.z, 
                    bodyPosMass.w, 
                    softeningSquared );
                acc[0] += fx;
                acc[1] += fy;
                acc[2] += fz;
            }
            __syncthreads();
        }
        force[3*m+0] += acc[0];
        force[3*m+1] += acc[1];
        force[3*m+2] += acc[2];
    }
}

#endif
