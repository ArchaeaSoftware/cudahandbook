/*
 *
 * l2Persistence.cu
 *
 * Microbenchmark for the L2 persisting-access control added in the Ampere
 * architecture (cudaAccessPolicyWindow). A gather kernel reads a small hot
 * lookup table many times per output element while streaming a large index
 * array in and a large output array out; the streaming traffic is what
 * evicts the hot table from L2 between reuses. Marking the table's address
 * range as *persisting* on the stream reserves L2 lines for it, so the
 * repeated lookups hit L2 instead of returning to device memory.
 *
 * The kernel is run twice on the same stream -- once with the ordinary L2
 * policy, once with a persisting access-policy window over the table -- and
 * the two times are compared. Both runs compute the same result, which is
 * checksummed to confirm the policy changes performance and not output.
 *
 * Build with the top-level CMake build. Requires: SM 8.0 or higher
 * (reports "persisting L2 unsupported" and exits cleanly below that).
 *
 * Copyright (c) 2013-2026, Archaea Software, LLC.
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

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "chError.h"

// Data array: 8M int32 = 32 MB, far larger than the 2.25 MB L2, so it cannot
// be cached as a whole. A 1.5 MB hot region at its front (<= the RTX 3060's
// 1.55 MB persisting set-aside) receives most of the accesses; the rest are
// scattered across the whole array to keep L2 under streaming pressure.
const uint32_t DATA_N = 8u * 1024u * 1024u;
const uint32_t HOT_N  = 393216;             // 1.5 MB hot region
const uint32_t HOT_PERCENT = 90;            // share of gathers hitting it

// Number of gathers (index and output arrays are this many int32 = 256 MB).
const uint32_t N = 64u * 1024u * 1024u;

__global__ void
gather( const int *__restrict__ data, const uint32_t *__restrict__ idx,
        int *__restrict__ out, uint32_t n )
{
    for ( uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
                   i < n;
                   i += blockDim.x*gridDim.x ) {
        out[i] = data[idx[i]];
    }
}

// Run the gather nTrials times on the given stream, returning the best time
// in milliseconds and an XOR checksum of the output in *checksum.
static float
timeGather( cudaStream_t stream, const int *data, const uint32_t *idx, int *out,
            int *hOut, int nBlocks, int nThreads, int nTrials,
            uint32_t *checksum )
{
    cudaError_t status_cudart;
    cudaEvent_t evStart = nullptr, evStop = nullptr;
    float msBest = 1e30f;
    uint32_t x = 0;

    cuda(EventCreate( &evStart ));
    cuda(EventCreate( &evStop ));

    for ( int t = 0; t < nTrials; t++ ) {
        float ms;
        cuda(EventRecord( evStart, stream ));
        gather<<<nBlocks, nThreads, 0, stream>>>( data, idx, out, N );
        cuda(EventRecord( evStop, stream ));
        cuda(EventSynchronize( evStop ));
        cuda(EventElapsedTime( &ms, evStart, evStop ));
        if ( ms < msBest ) {
            msBest = ms;
        }
    }

    cuda(Memcpy( hOut, out, (size_t) N*sizeof(int), cudaMemcpyDeviceToHost ));
    for ( uint32_t i = 0; i < N; i++ ) {
        x ^= (uint32_t) hOut[i];
    }
    *checksum = x;

Error_cudart:
    if ( evStart ) cudaEventDestroy( evStart );
    if ( evStop  ) cudaEventDestroy( evStop );
    return msBest;
}

int
main( int argc, char *argv[] )
{
    cudaError_t status_cudart;
    int ret = 1;
    int *dData = nullptr, *dOut = nullptr, *hData = nullptr, *hOut = nullptr;
    uint32_t *dIdx = nullptr, *hIdx = nullptr;
    cudaStream_t stream = nullptr;
    cudaDeviceProp prop;
    int device = 0, nSMs = 0, nBlocks = 0;
    const int nThreads = 256, nTrials = 6;
    const size_t dataBytes = (size_t) DATA_N * sizeof(int);
    const size_t idxBytes  = (size_t) N * sizeof(uint32_t);
    const size_t outBytes  = (size_t) N * sizeof(int);
    const size_t hotBytes  = (size_t) HOT_N * sizeof(int);
    float msBase = 0.0f, msPersist = 0.0f;
    uint32_t sumBase = 0, sumPersist = 0, rng = 12345u;
    cudaStreamAttrValue attr = {}, zero = {};

    cuda(GetDevice( &device ));
    cuda(GetDeviceProperties( &prop, device ));
    if ( prop.persistingL2CacheMaxSize == 0 ) {
        printf( "%s: persisting L2 unsupported (needs SM 8.0+); skipping.\n",
                prop.name );
        return 0;
    }
    nSMs = prop.multiProcessorCount;
    nBlocks = nSMs * 32;

    cuda(Malloc( &dData, dataBytes ));
    cuda(Malloc( &dIdx, idxBytes ));
    cuda(Malloc( &dOut, outBytes ));
    hData = (int *) malloc( dataBytes );
    hIdx = (uint32_t *) malloc( idxBytes );
    hOut = (int *) malloc( outBytes );
    if ( ! hData || ! hIdx || ! hOut ) {
        fprintf( stderr, "host allocation failed\n" );
        goto Error_cudart;
    }
    for ( uint32_t i = 0; i < DATA_N; i++ ) {
        hData[i] = (int) (i*2654435761u);
    }
    // HOT_PERCENT of the gathers land in the hot region [0, HOT_N); the rest
    // scatter across the whole 32 MB array, keeping L2 under streaming load.
    for ( uint32_t i = 0; i < N; i++ ) {
        rng = rng*1664525u + 1013904223u;
        uint32_t pick = (rng >> 8) % 100u;
        rng = rng*1664525u + 1013904223u;
        uint32_t v = rng >> 4;
        hIdx[i] = (pick < HOT_PERCENT) ? (v % HOT_N) : (v % DATA_N);
    }
    cuda(Memcpy( dData, hData, dataBytes, cudaMemcpyHostToDevice ));
    cuda(Memcpy( dIdx, hIdx, idxBytes,   cudaMemcpyHostToDevice ));
    cuda(StreamCreate( &stream ));

    printf( "%s: %d SMs, L2 %.2f MB, persisting set-aside max %.2f MB\n",
            prop.name, nSMs, prop.l2CacheSize/1048576.0,
            prop.persistingL2CacheMaxSize/1048576.0 );
    printf( "data %.0f MB, hot region %.2f MB (%u%% of gathers), "
            "%.0f M gathers, best of %d\n\n",
            dataBytes/1048576.0, hotBytes/1048576.0, HOT_PERCENT,
            N/1048576.0, nTrials );

    // Baseline: default L2 policy, no persisting window.
    cuda(CtxResetPersistingL2Cache());
    msBase = timeGather( stream, dData, dIdx, dOut, hOut,
                         nBlocks, nThreads, nTrials, &sumBase );

    // Reserve L2 for persisting accesses and mark the hot region's range
    // persisting on the stream; accesses outside it use the default policy.
    cuda(DeviceSetLimit( cudaLimitPersistingL2CacheSize,
                         prop.persistingL2CacheMaxSize ));
    attr.accessPolicyWindow.base_ptr  = dData;
    attr.accessPolicyWindow.num_bytes = hotBytes;   // <= accessPolicyMaxWindowSize
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cuda(StreamSetAttribute( stream, cudaStreamAttributeAccessPolicyWindow,
                             &attr ));

    msPersist = timeGather( stream, dData, dIdx, dOut, hOut,
                            nBlocks, nThreads, nTrials, &sumPersist );

    // Release the persisting lines and the set-aside.
    cuda(StreamSetAttribute( stream, cudaStreamAttributeAccessPolicyWindow,
                             &zero ));
    cuda(CtxResetPersistingL2Cache());
    cuda(DeviceSetLimit( cudaLimitPersistingL2CacheSize, 0 ));

    if ( sumBase != sumPersist ) {
        fprintf( stderr, "verification failed: checksums differ "
                 "(0x%08x vs 0x%08x)\n", sumBase, sumPersist );
        goto Error_cudart;
    }

    printf( "  %-24s %8.3f ms  (%.2fx)\n", "default L2 policy",
            msBase, 1.0 );
    printf( "  %-24s %8.3f ms  (%.2fx)\n", "persisting table",
            msPersist, msBase / msPersist );

    ret = 0;

Error_cudart:
    if ( stream ) cudaStreamDestroy( stream );
    if ( dData ) cudaFree( dData );
    if ( dIdx ) cudaFree( dIdx );
    if ( dOut ) cudaFree( dOut );
    free( hData ); free( hIdx ); free( hOut );
    return ret;
}
