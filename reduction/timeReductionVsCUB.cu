/*
 * timeReductionVsCUB.cu
 *
 * Same-card throughput comparison of the Chapter 12 reductions against CUB
 * and Thrust. Sum of N 32-bit ints -> one int. Reports ms, effective GB/s
 * (N*4 bytes read), Gelem/s, and a correctness check against a serial CPU
 * sum.
 *
 * Build (from reduction/ in the repo):
 *   nvcc -O3 -arch=sm_86 -I ../chLib timeReductionVsCUB.cu -o timeReductionVsCUB
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <chError.h>

#include "reductionWarpShuffle.cuh"      // Reduction2  (shuffle, two pass)
#include "reduction5Atomics.cuh"         // Reduction5  (global atomic, one pass)
#include "reductionWarpShuffleCG.cuh"    // ReductionCG (cg::reduce block reduce)
#include "reductionVectorized.cuh"       // ReductionVector (int4 + atomic)

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

static long long g_ref = 0;
static size_t g_N = 0;

static void report( const char *name, int result, double ms )
{
    double gbps  = ((double) g_N * sizeof(int)) / (ms * 1e-3) / 1e9;
    double gelem = g_N / (ms * 1e-3) / 1e9;
    bool ok = ( (long long) result == g_ref );
    printf( "  %-34s %8.3f %10.1f %10.2f   %s\n",
            name, ms, gbps, gelem, ok ? "ok" : "FAIL" );
}

// Time a two-argument (answer, partial) reduction and read back its int result.
typedef void (*pfnReduction)(int *out, int *partial, const int *in, size_t N, int nb, int nt);

static void timeOurs( const char *name, pfnReduction pfn,
                      int *d_ans, int *d_partial, const int *d_in,
                      size_t N, int nb, int nt, int iters )
{
    double ms = time_ms([&]{ pfn( d_ans, d_partial, d_in, N, nb, nt ); }, iters);
    int h = 0;
    CHECK( cudaMemcpy(&h, d_ans, sizeof(int), cudaMemcpyDeviceToHost) );
    report( name, h, ms );
}

int
main( int argc, char *argv[] )
{
    size_t N  = ( argc > 1 ) ? (size_t) atoll(argv[1]) : (64u << 20);   // 64M
    int    nt = ( argc > 2 ) ? atoi(argv[2]) : 256;
    int    nb = ( argc > 3 ) ? atoi(argv[3]) : 1024;
    const int iters = 100;
    g_N = N;

    cudaDeviceProp prop; int dev = 0;
    CHECK( cudaGetDevice(&dev) ); CHECK( cudaGetDeviceProperties(&prop, dev) );
    printf( "GPU: %s (sm_%d%d)   N=%zu (%.0fM)   blocks=%d threads=%d   iters=%d\n\n",
            prop.name, prop.major, prop.minor, N, N / 1048576.0, nb, nt, iters );

    int *h_in = (int *) malloc( N * sizeof(int) );
    srand( 1 );
    { long long s = 0; for ( size_t i = 0; i < N; i++ ) { int v = rand() & 1; h_in[i] = v; s += v; } g_ref = s; }

    int *d_in = 0, *d_ans = 0, *d_partial = 0;
    CHECK( cudaMalloc(&d_in,      N  * sizeof(int)) );
    CHECK( cudaMalloc(&d_ans,          sizeof(int)) );
    CHECK( cudaMalloc(&d_partial, nb * sizeof(int)) );
    CHECK( cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice) );

    printf( "  %-34s %8s %10s %10s   %s\n",
            "implementation", "ms", "GB/s", "Gelem/s", "correct" );
    printf( "  %-34s %8s %10s %10s   %s\n",
            "----------------------------------", "-------", "--------", "-------", "-------" );

    // --- our reductions ------------------------------------------------------
    timeOurs( "warp shuffle (two pass)",   Reduction2,      d_ans, d_partial, d_in, N, nb, nt, iters );
    timeOurs( "global atomic (one pass)",  Reduction5,      d_ans, d_partial, d_in, N, nb, nt, iters );
    timeOurs( "cg::reduce block (two pass)",ReductionCG,    d_ans, d_partial, d_in, N, nb, nt, iters );
    timeOurs( "vectorized int4 (one pass)",ReductionVector, d_ans, d_partial, d_in, N, nb, nt, iters );

    // --- libraries -----------------------------------------------------------
    {
        void *d_tmp = 0; size_t tmpb = 0;
        cub::DeviceReduce::Sum(d_tmp, tmpb, d_in, d_ans, (int) N);
        CHECK( cudaMalloc(&d_tmp, tmpb) );
        double ms = time_ms([&]{
            size_t tb = tmpb;
            cub::DeviceReduce::Sum(d_tmp, tb, d_in, d_ans, (int) N);
        }, iters);
        int h = 0; CHECK( cudaMemcpy(&h, d_ans, sizeof(int), cudaMemcpyDeviceToHost) );
        report( "CUB DeviceReduce::Sum", h, ms );
        cudaFree(d_tmp);
    }
    {
        thrust::device_ptr<int> pin(d_in);
        int result = 0;
        double ms = time_ms([&]{ result = thrust::reduce(pin, pin + N, 0); }, iters);
        report( "Thrust reduce", result, ms );
    }

    free(h_in);
    cudaFree(d_in); cudaFree(d_ans); cudaFree(d_partial);
    return 0;
}
