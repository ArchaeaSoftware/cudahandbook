/*
 *
 * scanWarp2.cuh
 *
 * Alternative implementation of warp scan.
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

#ifndef __SCAN_WARP2_CUH__
#define __SCAN_WARP2_CUH__

#if 0
/*
 * scanWarp - assumes no zero padding
 */
template<class T>
inline __device__ T 
scanWarp( volatile T *sPartials )
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if ( lane >=  1 ) sPartials[0] += sPartials[- 1];
    if ( lane >=  2 ) sPartials[0] += sPartials[- 2];
    if ( lane >=  4 ) sPartials[0] += sPartials[- 4];
    if ( lane >=  8 ) sPartials[0] += sPartials[- 8];
    if ( lane >= 16 ) sPartials[0] += sPartials[-16];
    return sPartials[0];
}
#endif

/*
 * scanWarp - bZeroPadded template parameter specifies
 *    whether to conditionally add based on the lane ID.
 *    If we can assume that sPartials[-1..-16] is 0,
 *    the routine takes fewer instructions.
 * idx is the base index of the warp to scan.
 */
template<class T, bool bZeroPadded>
inline __device__ T
scanWarp2( volatile T *sPartials )
{
    if ( bZeroPadded ) {
        T t = sPartials[0];
        sPartials[0] = t = t + sPartials[- 1];
        sPartials[0] = t = t + sPartials[- 2];
        sPartials[0] = t = t + sPartials[- 4];
        sPartials[0] = t = t + sPartials[- 8];
        sPartials[0] = t = t + sPartials[-16];
    }
    else {
        const int tid = threadIdx.x;
        const int lane = tid & 31;

        if ( lane >=  1 ) sPartials[0] += sPartials[- 1];
        if ( lane >=  2 ) sPartials[0] += sPartials[- 2];
        if ( lane >=  4 ) sPartials[0] += sPartials[- 4];
        if ( lane >=  8 ) sPartials[0] += sPartials[- 8];
        if ( lane >= 16 ) sPartials[0] += sPartials[-16];
    }
    return sPartials[0];
}

template<class T, bool bZeroPadded>
inline __device__ T
scanWarpExclusive2( volatile T *sPartials )
{
    if ( bZeroPadded ) {
        T t = sPartials[0];
        sPartials[0] = t = t + sPartials[- 1];
        sPartials[0] = t = t + sPartials[- 2];
        sPartials[0] = t = t + sPartials[- 4];
        sPartials[0] = t = t + sPartials[- 8];
        sPartials[0] = t = t + sPartials[-16];
    }
    else {
        const int tid = threadIdx.x;
        const int lane = tid & 31;

        if ( lane >=  1 ) sPartials[0] += sPartials[- 1];
        if ( lane >=  2 ) sPartials[0] += sPartials[- 2];
        if ( lane >=  4 ) sPartials[0] += sPartials[- 4];
        if ( lane >=  8 ) sPartials[0] += sPartials[- 8];
        if ( lane >= 16 ) sPartials[0] += sPartials[-16];
    }
    return (threadIdx.x&31) ? sPartials[-1] : 0;
}

#endif // __SCAN_WARP2_CUH__
