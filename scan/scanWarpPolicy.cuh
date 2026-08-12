/*
 *
 * scanWarpPolicy.cuh
 *
 * Warp- and block-level inclusive/exclusive scan, with the warp-scan
 * implementation selected at compile time by a POLICY TYPE. Each variant is a
 * distinct struct exposing a static inclusive() method; the scan algorithms are
 * single function templates instantiated per policy -- no runtime function
 * pointer and no if-constexpr switch. Adding a variant is adding a struct.
 *
 * (A function template cannot be partially specialized, so a policy type is the
 * idiomatic way to get "different instantiations of the same template" here.)
 *
 * Policies:
 *   WarpScanShared  -- Kogge-Stone in shared memory, Volta-clean (__syncwarp).
 *   WarpScanShuffle -- register-only, __shfl_up_sync (no shared traffic).
 * The pre-Volta unsynchronized shared scan (formerly scanWarp2) is intentionally
 * gone: it is slower here and unsafe under independent thread scheduling.
 *
 * Copyright (c) 2025-2026, Archaea Software, LLC.
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
#ifndef __SCAN_WARP_POLICY_CUH__
#define __SCAN_WARP_POLICY_CUH__

// ---- warp-scan policies: each is a distinct type carrying one implementation ----

struct WarpScanShared {                        // Kogge-Stone in shared, Volta-clean
    static constexpr const char *name = "WarpScanShared";
    template<class T>
    static inline __device__ T
    inclusive( T val, volatile T *scr )
    {
        const int lane = threadIdx.x & 31;
        scr[0] = val;
        T t = val;
        #pragma unroll
        for ( int offset = 1; offset < 32; offset <<= 1 ) {
            if ( lane >= offset ) t += scr[-offset];
            __syncwarp();
            scr[0] = t;
            __syncwarp();
        }
        return t;
    }
};

struct WarpScanShuffle {                        // register-only, __shfl_up_sync
    static constexpr const char *name = "WarpScanShuffle";
    template<class T>
    static inline __device__ T
    inclusive( T val, volatile T * /*unused*/ )
    {
        const int lane = threadIdx.x & 31;
        #pragma unroll
        for ( int offset = 1; offset < 32; offset <<= 1 ) {
            T n = __shfl_up_sync( 0xffffffffu, val, offset );
            if ( lane >= offset ) val += n;
        }
        return val;
    }
};

// ---- one function template each, instantiated per policy type ----

template<class WarpPolicy, class T>
inline __device__ T
warpScanInclusive( T val, volatile T *scr )
{
    return WarpPolicy::template inclusive<T>( val, scr );
}

template<class WarpPolicy, class T>
inline __device__ T
warpScanExclusive( T val, volatile T *scr )
{
    return warpScanInclusive<WarpPolicy,T>( val, scr ) - val;
}

// Block-wide inclusive scan, composed on the SAME policy -- the warp scan inlines
// (no function pointer). scr: blockDim ints of shared, warp-contiguous so the
// shared policy's neighbour reads stay within a warp's own run.
template<class WarpPolicy, class T>
inline __device__ T
blockScanInclusive( T val, volatile T *scr )
{
    const int tid = threadIdx.x, lane = tid & 31, warpid = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    __shared__ T warpTotals[32];

    T v = WarpPolicy::template inclusive<T>( val, scr + tid );
    if ( 31 == lane ) warpTotals[warpid] = v;
    __syncthreads();
    if ( 0 == warpid ) {                        // exclusive scan of the per-warp totals
        T w = ( lane < nwarps ) ? warpTotals[lane] : (T)0, inc = w;
        #pragma unroll
        for ( int offset = 1; offset < 32; offset <<= 1 ) {
            T n = __shfl_up_sync( 0xffffffffu, inc, offset );
            if ( lane >= offset ) inc += n;
        }
        warpTotals[lane] = inc - w;
    }
    __syncthreads();
    return v + warpTotals[warpid];
}

#endif // __SCAN_WARP_POLICY_CUH__
