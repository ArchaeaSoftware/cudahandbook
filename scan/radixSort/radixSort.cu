/*
 *
 * radixSort.cu
 *
 * Microdemo and microbenchmark of Radix Sort. Two implementations: a
 * host-scan reference (GPU histogram, host scan and scatter) that shows
 * the algorithm, and a fully GPU-resident stable LSD radix sort (per-tile
 * local sort, device-wide scan, scatter) that keeps all three phases on
 * the GPU. See Chapter 16, "Histograms and Radix Sort."
 *
 * Build with: nvcc -I ../chLib <options> radixSort.cu
 * Requires: SM 2.0 or higher (global/shared atomics).
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

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


#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

#include <assert.h>
#include <stdint.h>
#include <chrono>
#include <chError.h>


#define NUM_THREADS 64

template<const int b>
__global__ void
RadixHistogram_device( int *dptrHistogram, const uint32_t *in, size_t N, int shift, int mask )
{
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        int index = (in[i] & mask) >> shift;
        atomicAdd( dptrHistogram+index, 1 );
    }
#if 0
    const int cBuckets = 1<<b;
    __shared__ uint8_t sharedHistogram[NUM_THREADS][cBuckets];

    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
        int index = (in[i] & mask) >> shift;
        if ( 0 == ++sharedHistogram[threadIdx.x][index] ) {
            atomicAdd( dptrHistogram+index, 256 );
        }
    }
    __syncthreads();
    for ( int i = 0; i < cBuckets; i++ ) {
        if ( sharedHistogram[threadIdx.x][i] ) {
            atomicAdd( dptrHistogram+i, sharedHistogram[threadIdx.x][i] );
        }
    }
#endif
}

template<const int b>
void
RadixHistogram( int *dptrHistogram, const uint32_t *in, size_t N, int shift, int mask, int cBlocks, int cThreads )
{
    RadixHistogram_device<b><<<cBlocks,cThreads>>>( dptrHistogram, in, N, shift, mask );
}


template<const int b>
bool
RadixPass( uint32_t *out, const uint32_t *in, size_t N, int shift, int mask )
{
    bool ret = false;
    cudaError_t status_cudart;
    const int numCounts = 1<<b;
    int counts[numCounts];
    memset( counts, 0, sizeof(counts) );

uint32_t *gpuIn = 0;
int *gpuHistogram = 0;
int *cpuHistogram = 0;
cuda(Malloc( &gpuIn, N*sizeof(uint32_t) ) );
cuda(Memcpy( gpuIn, in, N*sizeof(uint32_t), cudaMemcpyHostToDevice ) );
cuda(Malloc( &gpuHistogram, (1<<b)*sizeof(int) ) );
cuda(Memset( gpuHistogram, 0, (1<<b)*sizeof(int) ) );
cpuHistogram = (int *) malloc( (1<<b)*sizeof(int) );
if ( ! cpuHistogram ) {
    status_cudart = cudaErrorMemoryAllocation;
    goto Error_cudart;
}

RadixHistogram<b>( gpuHistogram, gpuIn, N, shift, mask, 1500, 512 );
cuda(Memcpy( cpuHistogram, gpuHistogram, (1<<b)*sizeof(int), cudaMemcpyDeviceToHost ) );


    for ( size_t i = 0; i < N; i++ ) {
        uint32_t value = in[i];
        int index = (value & mask) >> shift;
        counts[index] += 1;
    }

for ( int j = 0; j < (1<<b); j++ ) {
    assert ( counts[j] == cpuHistogram[j] );
}

    //
    // compute exclusive scan of counts
    //
    {
        int sum = 0;
        for ( int i = 0; i < numCounts; i++ ) { 
            int temp = counts[i];
            counts[i] = sum;
            sum += temp;
        }
    }

    //
    // scatter each input to the correct output
    //
    for ( size_t i = 0; i < N; i++ ) {
        uint32_t value = in[i];
        int index = (value & mask) >> shift;
        out[ counts[index] ] = value;
        counts[index] += 1;
    }
    ret = true;
Error_cudart:
    cudaFree( gpuIn );
    cudaFree( gpuHistogram );
    free( cpuHistogram );
    
    return ret;
}

template<const int b>
uint32_t *
RadixSort( uint32_t *out[2], const uint32_t *in, size_t N )
{
    int shift = 0;
    int mask = (1<<b)-1;

    //
    // index of output array, ping-pongs between 0 and 1.
    //
    int outIndex = 0;

    RadixPass<b>( out[outIndex], in, N, shift, mask );
    while ( mask ) {
        outIndex = 1 - outIndex;
        shift += 1;
        mask <<= 1;
        RadixPass<b>( out[outIndex], out[1-outIndex], N, shift, mask );
    }
    return out[outIndex];

}

bool
TestSort( float *et, uint32_t *(*pfnSort)( uint32_t *[2], const uint32_t *, size_t ), size_t N, uint32_t mask = 0xffffffff )
{
    std::chrono::steady_clock::time_point start, stop;
    bool ret = false;
    uint32_t *sortInput = new uint32_t[ N ];
    uint32_t *sortOutput[2];
    uint32_t *radixSortedArray = 0;
    std::vector<uint32_t> sortedOutput( N );
    sortOutput[0] = new uint32_t[ N ];
    sortOutput[1] = new uint32_t[ N ];

    if ( 0 == sortInput || 
         0 == sortOutput[0] ||
         0 == sortOutput[1] ) {
        goto Error_cudart;
    }

    for ( int i = 0; i < N; i++ ) {
        sortedOutput[i] = sortInput[i] = (rand()|(rand()<<16)) & mask;
    }

    {
        std::sort( sortedOutput.begin(), sortedOutput.end() );
    }

    start = std::chrono::steady_clock::now();

    //
    // RadixSort returns sortOutput[0] or sortOutput[1],
    // depending on where it wound up in the ping-pong
    // between output arrays.
    //
    radixSortedArray = pfnSort( sortOutput, sortInput, N );

    stop = std::chrono::steady_clock::now();
    *et = std::chrono::duration<double>(stop - start).count();

    for ( size_t i = 0; i < N; i++ ) {
        if ( radixSortedArray[i] != sortedOutput[i] ) {
#ifdef _WIN32
            __debugbreak();
#endif
            goto Error_cudart;
        }
    }
    ret = true;
Error_cudart:
    delete[] sortInput;
    delete[] sortOutput[0];
    delete[] sortOutput[1];
    return ret;
}

//
// =====================================================================
// Fully GPU-resident, stable LSD radix sort.
//
// Each pass sorts the keys by one b-bit digit in three GPU phases:
//   1. RadixLocalSort - each threadblock stably sorts its own tile of
//      keys by the digit (b successive one-bit splits) and records the
//      tile's per-digit counts.
//   2. a device-wide scan of the digit-major count matrix gives every
//      (tile, digit) pair the base offset of its keys in the output.
//   3. RadixScatter - each tile writes its locally-sorted keys to
//      base + rank-within-the-tile's-digit-run.
// The passes ping-pong between two buffers.  Keys are treated as
// uint32_t, so their numeric ordering matches a std::sort of the input.
// =====================================================================
//

// One key per thread; a tile is one threadblock's worth of keys.
#define RADIX_TILE 256

//
// Inclusive scan of RADIX_TILE ints in shared memory (Hillis-Steele).
// Every thread of the block participates.
//
__device__ void
inclusiveScanBlock( volatile int *s )
{
    const int tid = threadIdx.x;
    for ( int off = 1; off < RADIX_TILE; off <<= 1 ) {
        int v = (tid >= off) ? s[tid-off] : 0;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }
}

// --- device-wide scan of an int array (stands in for Chapter 13) -----

__global__ void
reduceBlocks( int *blockSums, const int *in, size_t M )
{
    __shared__ int s[RADIX_TILE];
    const int tid = threadIdx.x;
    size_t i = (size_t) blockIdx.x * RADIX_TILE + tid;
    s[tid] = (i < M) ? in[i] : 0;
    __syncthreads();
    for ( int off = RADIX_TILE>>1; off > 0; off >>= 1 ) {
        if ( tid < off ) s[tid] += s[tid+off];
        __syncthreads();
    }
    if ( tid == 0 ) blockSums[blockIdx.x] = s[0];
}

// out[i] = inclusiveScan(in)[i] + (blockBase ? blockBase[block] : 0)
__global__ void
scanBlocksWithBase( int *out, const int *in, const int *blockBase, size_t M )
{
    __shared__ int s[RADIX_TILE];
    const int tid = threadIdx.x;
    size_t i = (size_t) blockIdx.x * RADIX_TILE + tid;
    s[tid] = (i < M) ? in[i] : 0;
    __syncthreads();
    inclusiveScanBlock( s );
    if ( i < M ) out[i] = s[tid] + (blockBase ? blockBase[blockIdx.x] : 0);
}

__global__ void
shiftToExclusive( int *outExcl, const int *incl, size_t n )
{
    size_t i = (size_t) blockIdx.x * RADIX_TILE + threadIdx.x;
    if ( i < n ) outExcl[i] = (i>0) ? incl[i-1] : 0;
}

static void devScanExclusive( int *out, const int *in, size_t n, int *scratch, cudaStream_t stream );

// Inclusive scan of in[0..M) -> out[0..M).  scratch holds working buffers.
static void
devScanInclusive( int *out, const int *in, size_t M, int *scratch, cudaStream_t stream )
{
    int nb = (int)((M + RADIX_TILE - 1) / RADIX_TILE);
    if ( nb == 1 ) {
        scanBlocksWithBase<<<1, RADIX_TILE, 0, stream>>>( out, in, 0, M );
        return;
    }
    int *blockSums     = scratch;          // nb
    int *blockSumsExcl = scratch + nb;     // nb
    int *rest          = scratch + 2*nb;
    reduceBlocks<<<nb, RADIX_TILE, 0, stream>>>( blockSums, in, M );
    devScanExclusive( blockSumsExcl, blockSums, nb, rest, stream );
    scanBlocksWithBase<<<nb, RADIX_TILE, 0, stream>>>( out, in, blockSumsExcl, M );
}

// Exclusive scan of in[0..n) -> out[0..n).
static void
devScanExclusive( int *out, const int *in, size_t n, int *scratch, cudaStream_t stream )
{
    int *incl = scratch;                   // n
    int *rest = scratch + n;
    devScanInclusive( incl, in, n, rest, stream );
    int gb = (int)((n + RADIX_TILE - 1) / RADIX_TILE);
    shiftToExclusive<<<gb, RADIX_TILE, 0, stream>>>( out, incl, n );
}

//
// Phase 1: stable local sort of each tile by the b-bit digit, plus the
// tile's per-digit histogram (digit-major: blockHist[digit*numTiles+tile]).
//
template<int b>
__global__ void
RadixLocalSort(
    uint32_t *sortedKeys, int *blockHist,
    const uint32_t *in, size_t N, int shift, int numTiles )
{
    const int NUM_DIGITS = 1 << b;
    const uint32_t mask = NUM_DIGITS - 1;
    const int tid  = threadIdx.x;
    const int tile = blockIdx.x;
    const size_t base = (size_t) tile * RADIX_TILE;
    const int valid = (base + RADIX_TILE <= N) ? RADIX_TILE : (int)(N - base);

    __shared__ uint32_t s[RADIX_TILE];
    __shared__ uint32_t sTmp[RADIX_TILE];
    __shared__ int      sScan[RADIX_TILE];
    __shared__ int      sHist[1<<b];

    // Pad the last tile with 0xFFFFFFFF so padding sorts to the end.
    s[tid] = (tid < valid) ? in[base+tid] : 0xFFFFFFFFu;
    __syncthreads();

    for ( int bit = 0; bit < b; bit++ ) {
        int bitpos = shift + bit;
        int flag = (int)((s[tid] >> bitpos) & 1);
        sScan[tid] = 1 - flag;                 // 1 marks a 0-bit ("false")
        __syncthreads();
        inclusiveScanBlock( sScan );
        int totalFalses = sScan[RADIX_TILE-1];
        int f = sScan[tid] - (1 - flag);       // exclusive # falses before tid
        int dest = flag ? (totalFalses + tid - f) : f;
        __syncthreads();
        sTmp[dest] = s[tid];
        __syncthreads();
        s[tid] = sTmp[tid];
        __syncthreads();
    }

    if ( tid < NUM_DIGITS ) sHist[tid] = 0;
    __syncthreads();
    if ( tid < valid ) atomicAdd( &sHist[(s[tid] >> shift) & mask], 1 );
    __syncthreads();

    sortedKeys[base+tid] = s[tid];
    if ( tid < NUM_DIGITS ) blockHist[tid*numTiles + tile] = sHist[tid];
}

//
// Phase 3: scatter each tile's locally-sorted keys to their final slots.
//
template<int b>
__global__ void
RadixScatter(
    uint32_t *out, const uint32_t *sortedKeys,
    const int *scanIncl, const int *blockHist,
    size_t N, int shift, int numTiles )
{
    const int NUM_DIGITS = 1 << b;
    const uint32_t mask = NUM_DIGITS - 1;
    const int tid  = threadIdx.x;
    const int tile = blockIdx.x;
    const size_t base = (size_t) tile * RADIX_TILE;
    const int valid = (base + RADIX_TILE <= N) ? RADIX_TILE : (int)(N - base);

    __shared__ int digitStart[1<<b];

    uint32_t key = sortedKeys[base+tid];
    int d = (int)((key >> shift) & mask);

    if ( tid < valid ) {
        int prevd = (tid>0) ? (int)((sortedKeys[base+tid-1] >> shift) & mask) : -1;
        if ( d != prevd ) digitStart[d] = tid;   // start of this digit's run
    }
    __syncthreads();

    if ( tid < valid ) {
        int gbase = scanIncl[d*numTiles+tile] - blockHist[d*numTiles+tile];
        out[gbase + (tid - digitStart[d])] = key;
    }
}

//
// Ping-pong over 32/b digit passes; the sorted result ends up in dOut.
//
template<int b>
static void
RadixSortGPU(
    uint32_t *dOut, uint32_t *dIn, uint32_t *dSorted,
    int *dBlockHist, int *dScanIncl, int *dScratch,
    size_t N, int numTiles, cudaStream_t stream = 0 )
{
    const int NUM_DIGITS = 1 << b;
    const int numPasses  = 32 / b;
    const size_t M = (size_t) numTiles * NUM_DIGITS;

    uint32_t *src = dIn, *dst = dOut;
    for ( int pass = 0; pass < numPasses; pass++ ) {
        int shift = pass * b;
        RadixLocalSort<b><<<numTiles, RADIX_TILE, 0, stream>>>(
            dSorted, dBlockHist, src, N, shift, numTiles );
        devScanInclusive( dScanIncl, dBlockHist, M, dScratch, stream );
        RadixScatter<b><<<numTiles, RADIX_TILE, 0, stream>>>(
            dst, dSorted, dScanIncl, dBlockHist, N, shift, numTiles );
        uint32_t *t = src; src = dst; dst = t;
    }
    if ( src != dOut )
        cudaMemcpyAsync( dOut, src, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream );
}

template<int b>
bool
TestSortGPU( float *et, size_t N )
{
    const int NUM_DIGITS = 1 << b;
    int numTiles   = (int)((N + RADIX_TILE - 1) / RADIX_TILE);
    size_t M       = (size_t) numTiles * NUM_DIGITS;
    size_t padded  = (size_t) numTiles * RADIX_TILE;
    bool ret = false;
    cudaError_t status_cudart;
    std::chrono::steady_clock::time_point start, stop;

    std::vector<uint32_t> in( N ), ref( N ), got( N );
    for ( size_t i = 0; i < N; i++ )
        ref[i] = in[i] = ((uint32_t) rand()) | ((uint32_t) rand() << 16);
    std::sort( ref.begin(), ref.end() );

    uint32_t *dIn=0,*dOut=0,*dSorted=0; int *dHist=0,*dScan=0,*dScratch=0;
    cuda(Malloc( &dIn,      N*sizeof(uint32_t) ));
    cuda(Malloc( &dOut,     N*sizeof(uint32_t) ));
    cuda(Malloc( &dSorted,  padded*sizeof(uint32_t) ));
    cuda(Malloc( &dHist,    M*sizeof(int) ));
    cuda(Malloc( &dScan,    M*sizeof(int) ));
    cuda(Malloc( &dScratch, (M+4096)*sizeof(int) ));
    cuda(Memcpy( dIn, in.data(), N*sizeof(uint32_t), cudaMemcpyHostToDevice ));

    start = std::chrono::steady_clock::now();
    RadixSortGPU<b>( dOut, dIn, dSorted, dHist, dScan, dScratch, N, numTiles );
    cuda(DeviceSynchronize());
    stop = std::chrono::steady_clock::now();
    *et = std::chrono::duration<double>(stop - start).count();

    cuda(Memcpy( got.data(), dOut, N*sizeof(uint32_t), cudaMemcpyDeviceToHost ));
    ret = std::equal( got.begin(), got.end(), ref.begin() );
Error_cudart:
    cudaFree(dIn); cudaFree(dOut); cudaFree(dSorted);
    cudaFree(dHist); cudaFree(dScan); cudaFree(dScratch);
    return ret;
}

//
// Time the fully GPU-resident sort launched directly versus captured once
// into a CUDA graph and replayed. The sort is a fixed sequence of kernel
// launches, so it captures cleanly; a graph collapses the whole sequence
// into one cudaGraphLaunch(). Graphs help when per-launch CPU overhead is
// large relative to the kernel work -- small N -- not when the kernels
// dominate. Both timings exclude the per-iteration input reset.
//
template<int b>
static void
benchGraph( size_t N )
{
    const int NUM_DIGITS = 1 << b;
    int numTiles  = (int)((N + RADIX_TILE - 1) / RADIX_TILE);
    size_t M      = (size_t) numTiles * NUM_DIGITS;
    size_t padded = (size_t) numTiles * RADIX_TILE;
    const int nTrials = 50;

    std::vector<uint32_t> in( N ), ref( N ), got( N );
    for ( size_t i = 0; i < N; i++ )
        ref[i] = in[i] = ((uint32_t) rand()) | ((uint32_t) rand() << 16);
    std::sort( ref.begin(), ref.end() );

    uint32_t *dOrig=0,*dIn=0,*dOut=0,*dSorted=0; int *dHist=0,*dScan=0,*dScr=0;
    cudaMalloc( &dOrig,   N*sizeof(uint32_t) );
    cudaMalloc( &dIn,     N*sizeof(uint32_t) );
    cudaMalloc( &dOut,    N*sizeof(uint32_t) );
    cudaMalloc( &dSorted, padded*sizeof(uint32_t) );
    cudaMalloc( &dHist,   M*sizeof(int) );
    cudaMalloc( &dScan,   M*sizeof(int) );
    cudaMalloc( &dScr,    (M+4096)*sizeof(int) );
    cudaMemcpy( dOrig, in.data(), N*sizeof(uint32_t), cudaMemcpyHostToDevice );

    cudaStream_t stream; cudaStreamCreate( &stream );
    cudaEvent_t e0, e1; cudaEventCreate( &e0 ); cudaEventCreate( &e1 );

    // Direct: the host loop issues every kernel launch itself.
    float msDirect = 1e30f;
    for ( int t = 0; t < nTrials; t++ ) {
        cudaMemcpyAsync( dIn, dOrig, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream );
        cudaEventRecord( e0, stream );
        RadixSortGPU<b>( dOut, dIn, dSorted, dHist, dScan, dScr, N, numTiles, stream );
        cudaEventRecord( e1, stream );
        cudaEventSynchronize( e1 );
        float ms; cudaEventElapsedTime( &ms, e0, e1 );
        if ( ms < msDirect ) msDirect = ms;
    }
    cudaMemcpy( got.data(), dOut, N*sizeof(uint32_t), cudaMemcpyDeviceToHost );
    bool okDirect = std::equal( got.begin(), got.end(), ref.begin() );

    // Capture the identical sequence once.
    cudaMemcpyAsync( dIn, dOrig, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream );
    cudaGraph_t graph; cudaGraphExec_t exec;
    cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal );
    RadixSortGPU<b>( dOut, dIn, dSorted, dHist, dScan, dScr, N, numTiles, stream );
    cudaError_t capErr = cudaStreamEndCapture( stream, &graph );
    if ( capErr != cudaSuccess ) {
        printf( "  b=%d N=%zu: capture failed (%s)\n", b, (size_t) N,
                cudaGetErrorString( capErr ) );
        return;
    }
    size_t nNodes = 0; cudaGraphGetNodes( graph, 0, &nNodes );
    cudaGraphInstantiate( &exec, graph, 0 );

    // Replay: one cudaGraphLaunch() stands in for the whole sequence.
    float msGraph = 1e30f;
    for ( int t = 0; t < nTrials; t++ ) {
        cudaMemcpyAsync( dIn, dOrig, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream );
        cudaEventRecord( e0, stream );
        cudaGraphLaunch( exec, stream );
        cudaEventRecord( e1, stream );
        cudaEventSynchronize( e1 );
        float ms; cudaEventElapsedTime( &ms, e0, e1 );
        if ( ms < msGraph ) msGraph = ms;
    }
    cudaMemcpy( got.data(), dOut, N*sizeof(uint32_t), cudaMemcpyDeviceToHost );
    bool okGraph = std::equal( got.begin(), got.end(), ref.begin() );

    printf( "  b=%d  N=%8zu  %3zu nodes   direct %8.3f ms   graph %8.3f ms   %5.2fx%s%s\n",
            b, (size_t) N, nNodes, msDirect, msGraph, msDirect/msGraph,
            okDirect?"":" [DIRECT WRONG]", okGraph?"":" [GRAPH WRONG]" );

    cudaGraphExecDestroy( exec ); cudaGraphDestroy( graph );
    cudaEventDestroy( e0 ); cudaEventDestroy( e1 ); cudaStreamDestroy( stream );
    cudaFree(dOrig); cudaFree(dIn); cudaFree(dOut); cudaFree(dSorted);
    cudaFree(dHist); cudaFree(dScan); cudaFree(dScr);
}

int
main()
{
    float ms;
    size_t N = 16*1048576;

#define TEST_VECTOR( fn, N, mask )  \
    if ( ! TestSort( &ms, fn, N, mask ) ) {  \
        printf( "%s (N=%d, mask=0x%x) FAILED\n", #fn, (int) N, (uint32_t) mask );  \
        exit(1);    \
    } \
    else { \
        printf( "%s (N=%d, mask=0x%x): %.2f Melements/s\n", #fn, (int) N, (uint32_t) mask, (double) (N/1e6)/(ms) ); \
    }

//    TEST_VECTOR( 32, 0xf );

    TEST_VECTOR( RadixSort<1>, N, 0xffffffff );
    TEST_VECTOR( RadixSort<2>, N, 0xffffffff );
    TEST_VECTOR( RadixSort<4>, N, 0xffffffff );

    TEST_VECTOR( RadixSort<1>, N, 0xf );
    TEST_VECTOR( RadixSort<2>, N, 0xf );
    TEST_VECTOR( RadixSort<4>, N, 0xf );

    TEST_VECTOR( RadixSort<1>, N, 0x1 );
    TEST_VECTOR( RadixSort<2>, N, 0x1 );
    TEST_VECTOR( RadixSort<4>, N, 0x1 );

    //
    // Fully GPU-resident radix sort.
    //
#define GPU_TEST( b, N )  \
    if ( ! TestSortGPU<b>( &ms, N ) ) {  \
        printf( "RadixSortGPU<%d> (N=%d) FAILED\n", b, (int) (N) );  \
        exit(1);  \
    } else { \
        printf( "RadixSortGPU<%d> (N=%d): %.2f Melements/s\n", \
                b, (int)(N), (double)((N)/1e6)/(ms) );  \
    }

    GPU_TEST( 1, N );
    GPU_TEST( 2, N );
    GPU_TEST( 4, N );
    GPU_TEST( 8, N );
    GPU_TEST( 4, 100003 );    // exercise a partial final tile

    //
    // Throughput as a function of the radix b (bits/digit -> 32/b passes),
    // at three sizes, to find the best b.
    //
    printf( "\nThroughput by radix b:\n" );
    printf( "-- N = 1048576 --\n" );
    GPU_TEST( 1, 1048576 ); GPU_TEST( 2, 1048576 );
    GPU_TEST( 4, 1048576 ); GPU_TEST( 8, 1048576 );
    printf( "-- N = 65536 --\n" );
    GPU_TEST( 1, 65536 ); GPU_TEST( 2, 65536 );
    GPU_TEST( 4, 65536 ); GPU_TEST( 8, 65536 );

    //
    // Does capturing the sort into a CUDA graph pay off? Sweep N from large
    // (kernel-bound) to small (launch-overhead-bound).
    //
    printf( "\nCUDA graph capture vs. direct launch (best of 50):\n" );
    benchGraph<4>( 16*1048576 );
    benchGraph<4>( 1048576 );
    benchGraph<4>( 65536 );
    benchGraph<4>( 4096 );
    benchGraph<4>( 1024 );
    benchGraph<8>( 65536 );
    benchGraph<8>( 1024 );
}
