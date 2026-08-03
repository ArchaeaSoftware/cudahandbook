/*
 * timeScanVsCUB.cu
 *
 * Same-card throughput comparison of the Chapter 13 scans against CUB and
 * Thrust. Inclusive sum of N 32-bit ints. Reports ms/scan, effective GB/s
 * (one read + one write = 2N*4 bytes) and Gelem/s, plus a correctness check
 * against a serial CPU reference.
 *
 * Build (from scan/int in the repo):
 *   nvcc -O3 -arch=sm_86 -I../../chLib timeScanVsCUB.cu -o timeScanVsCUB
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <chError.h>

#include "scanWarp.cuh"
#include "scanBlock.cuh"
#include "scanFan.cuh"
#include "scanReduceThenScan.cuh"
#include "scan2Level.cuh"
#include "scanDecoupledLookback.cuh"
#include "scanDecoupledLookback2.cuh"

#define CHECK(e) do { cudaError_t s_ = (e); if ( s_ != cudaSuccess ) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(s_), __FILE__, __LINE__); \
    exit(1); } } while (0)

template<class F>
static double time_ms( F fn, int iters )
{
    cudaEvent_t a, b;
    CHECK( cudaEventCreate(&a) ); CHECK( cudaEventCreate(&b) );
    fn();                                   // warm-up
    CHECK( cudaDeviceSynchronize() );
    CHECK( cudaEventRecord(a) );
    for ( int i = 0; i < iters; i++ ) fn();
    CHECK( cudaEventRecord(b) );
    CHECK( cudaEventSynchronize(b) );
    float ms = 0; CHECK( cudaEventElapsedTime(&ms, a, b) );
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

static int *g_ref = 0, *g_host = 0;
static size_t g_N = 0;

static void report( const char *name, int *d_out, double ms )
{
    CHECK( cudaMemcpy( g_host, d_out, g_N * sizeof(int), cudaMemcpyDeviceToHost ) );
    size_t mism = 0;
    for ( size_t i = 0; i < g_N; i++ ) if ( g_host[i] != g_ref[i] ) mism++;
    double gbps  = (2.0 * g_N * sizeof(int)) / (ms * 1e-3) / 1e9;
    double gelem = g_N / (ms * 1e-3) / 1e9;
    printf( "  %-34s %8.3f %10.1f %10.2f   %s\n",
            name, ms, gbps, gelem, mism ? "FAIL" : "ok" );
}

template<int IPT>
static void timeV2( int *d_out, const int *d_in, size_t N, int b, int iters )
{
    size_t tile = (size_t) b * IPT;
    unsigned numTiles = (unsigned)((N + tile - 1) / tile);
    scanStatus *st = 0; uint32_t *ctr = 0;
    CHECK( cudaMalloc(&st, numTiles * sizeof(scanStatus)) );
    CHECK( cudaMalloc(&ctr, sizeof(uint32_t)) );
    char name[64]; snprintf(name, sizeof(name), "decoupled ILP=%d (kernel only)", IPT);
    double ms = time_ms([&]{
        cudaMemset(st, 0, numTiles * sizeof(scanStatus));
        cudaMemset(ctr, 0, sizeof(uint32_t));
        scanDecoupledLookback2_kernel<int,IPT><<<numTiles, b, tile * sizeof(int)>>>(
            d_out, d_in, st, ctr, N);
    }, iters);
    report(name, d_out, ms);
    cudaFree(st); cudaFree(ctr);
}

int
main( int argc, char *argv[] )
{
    size_t N = ( argc > 1 ) ? (size_t) atoll(argv[1]) : (64u << 20);   // 64M
    int    b = ( argc > 2 ) ? atoi(argv[2]) : 256;
    const int iters = 50;
    g_N = N;

    cudaDeviceProp prop; int dev = 0;
    CHECK( cudaGetDevice(&dev) ); CHECK( cudaGetDeviceProperties(&prop, dev) );
    printf( "GPU: %s (sm_%d%d)   N=%zu (%.0fM)   blockSize=%d   iters=%d\n\n",
            prop.name, prop.major, prop.minor, N, N / 1048576.0, b, iters );

    int *h_in = (int *) malloc( N * sizeof(int) );
    g_ref     = (int *) malloc( N * sizeof(int) );
    g_host    = (int *) malloc( N * sizeof(int) );
    srand( 1 );
    for ( size_t i = 0; i < N; i++ ) h_in[i] = (rand() & 0xff) - 0x7f;
    { int s = 0; for ( size_t i = 0; i < N; i++ ) { s += h_in[i]; g_ref[i] = s; } }

    int *d_in = 0, *d_out = 0;
    CHECK( cudaMalloc(&d_in,  N * sizeof(int)) );
    CHECK( cudaMalloc(&d_out, N * sizeof(int)) );
    CHECK( cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice) );

    printf( "  %-34s %8s %10s %10s   %s\n",
            "implementation", "ms/scan", "GB/s", "Gelem/s", "correct" );
    printf( "  %-34s %8s %10s %10s   %s\n",
            "----------------------------------", "-------", "--------", "-------", "-------" );

    // --- our scans -----------------------------------------------------------
    report( "scan-then-fan (scanFan)",
            d_out, time_ms([&]{ scanFan<int>(d_out, d_in, N, b); }, iters) );
    report( "reduce-then-scan (scan2Level)",
            d_out, time_ms([&]{ scan2Level<int>(d_out, d_in, N, b); }, iters) );
    report( "decoupled look-back (as written)",
            d_out, time_ms([&]{ scanDecoupledLookback<int>(d_out, d_in, N, b); }, iters) );

    // decoupled variants, scratch pre-allocated (kernel only)
    {
        unsigned numTiles = (unsigned)((N + b - 1) / b);
        scanStatus *st = 0; uint32_t *ctr = 0;
        CHECK( cudaMalloc(&st,  numTiles * sizeof(scanStatus)) );
        CHECK( cudaMalloc(&ctr, sizeof(uint32_t)) );
        report( "decoupled look-back (kernel only)", d_out, time_ms([&]{
            cudaMemset(st, 0, numTiles * sizeof(scanStatus));
            cudaMemset(ctr, 0, sizeof(uint32_t));
            scanDecoupledLookback_kernel<int><<<numTiles, b, b * sizeof(int)>>>(
                d_out, d_in, st, ctr, N);
        }, iters) );
        cudaFree(st); cudaFree(ctr);
    }
    timeV2<4>(d_out, d_in, N, b, iters);
    timeV2<8>(d_out, d_in, N, b, iters);
    timeV2<16>(d_out, d_in, N, b, iters);

    // --- libraries -----------------------------------------------------------
    {
        void  *d_tmp = 0; size_t tmpb = 0;
        cub::DeviceScan::InclusiveSum(d_tmp, tmpb, d_in, d_out, N);
        CHECK( cudaMalloc(&d_tmp, tmpb) );
        report( "CUB DeviceScan::InclusiveSum", d_out, time_ms([&]{
            size_t tb = tmpb;
            cub::DeviceScan::InclusiveSum(d_tmp, tb, d_in, d_out, N);
        }, iters) );
        cudaFree(d_tmp);
    }
    {
        thrust::device_ptr<int> pin(d_in), pout(d_out);
        report( "Thrust inclusive_scan", d_out, time_ms([&]{
            thrust::inclusive_scan(pin, pin + N, pout);
        }, iters) );
    }

    free(h_in); free(g_ref); free(g_host);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
