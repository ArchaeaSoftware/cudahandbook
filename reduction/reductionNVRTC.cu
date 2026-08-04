/*
 *
 * reductionNVRTC.cu
 *
 * Runtime-generated reduction: the reduction is a transform_reduce over a
 * user-defined monoid whose pieces -- the accumulator type Acc, the
 * per-element load (transform), the combine operator, the identity, and an
 * optional captured Context -- are supplied as source text and compiled at
 * run time with NVRTC. This is the runtime twin of thrust::transform_reduce,
 * whose operator and type are fixed at compile time.
 *
 * The generated kernel is an election-based single pass (after the
 * threadFenceReduction SDK sample): every block reduces its slice with
 * combine and publishes its partial; one integer atomic elects the block
 * that finishes last; that block combines the partials into the result. The
 * only fixed-type atomic is the election counter -- always an unsigned int --
 * so Acc can be anything.
 *
 * The sample shows three things NVRTC buys that a compile-time library cannot:
 *   1. two different reductions (sum+sumsq, then a threshold count) generated
 *      in the same run, with no host recompile;
 *   2. the election counter provided two ways -- as a kernel argument, and as
 *      an NVRTC module __device__ global reached with cuModuleGetGlobal;
 *   3. a captured Context (a runtime threshold), which changes the generated
 *      kernel's *interface* -- so the host assembles a different launch
 *      argument list to match the signature NVRTC produced.
 *
 * Build (from reduction/ in the repo):
 *   nvcc -O3 -arch=sm_86 -I ../chLib reductionNVRTC.cu -lnvrtc -lcuda -o reductionNVRTC
 *
 * Copyright (c) 2011-2026, Archaea Software, LLC. All rights reserved.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <chError.h>       // chLib: reuse the cuda() and cu() error-check macros

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/count.h>

// cuda(), cu(), and nvrtc() all come from chError.h (chError_cuda.h): on
// failure each sets the enclosing function's status_<api> and jumps to its
// Error_<api>: label. nvrtc() is available because <nvrtc.h> is included above.

//
// The pieces the developer supplies. Everything else in the kernel is
// type-independent boilerplate. CtxDecl is optional: when present, load
// receives a const Ctx& and the kernel gains a const Ctx* parameter.
//
struct Monoid {
    const char *In;          // input element type
    const char *AccDecl;     // accumulator type      ("typedef ..." or "struct Acc {...};")
    const char *CtxDecl;     // captured context type (NULL/"" for none)
    const char *loadBody;    // In[,Ctx] -> Acc
    const char *combineBody; // (Acc,Acc) -> Acc
    const char *idBody;      // -> Acc
};

static bool hasCtx( const Monoid &m ) { return m.CtxDecl && m.CtxDecl[0]; }

//
// Assemble the kernel source. globalCounter selects whether the election
// counter is a module __device__ global or a kernel argument.
//
static std::string
buildSource( const Monoid &m, bool globalCounter )
{
    const bool ctx = hasCtx(m);
    std::string s;
    s += m.AccDecl; s += "\n";
    if ( ctx ) { s += m.CtxDecl; s += "\n"; }
    s += "typedef "; s += m.In; s += " In;\n\n";

    s += ctx ? "__device__ __forceinline__ Acc ld( In x, const Ctx &c )  { "
             : "__device__ __forceinline__ Acc ld( In x )                { ";
    s += m.loadBody; s += " }\n";
    s += "__device__ __forceinline__ Acc cmb( Acc a, Acc b )  { "; s += m.combineBody; s += " }\n";
    s += "__device__ __forceinline__ Acc idty()               { "; s += m.idBody;      s += " }\n\n";

    if ( globalCounter )
        s += "extern \"C\" __device__ unsigned int retirementCount = 0;\n\n";

    const char *counterParam = globalCounter ? ""                     : "unsigned int *retire, ";
    const char *ctxParam     = ctx           ? "const Ctx *ctx, "     : "";
    const char *counterAddr  = globalCounter ? "&retirementCount"     : "retire";
    const char *counterReset = globalCounter ? "retirementCount = 0;" : "*retire = 0;";
    const char *loadCall     = ctx           ? "ld( in[i], *ctx )"    : "ld( in[i] )";

    s += "extern \"C\" __global__ void\n";
    s += "reduce( Acc *out, Acc *partials, ";
    s += counterParam;
    s += ctxParam;
    s += "const In *in, unsigned long long N )\n";
    s += "{\n";
    s += "    extern __shared__ Acc sdata[];\n";
    s += "    __shared__ bool s_last;\n";
    s += "    const unsigned tid = threadIdx.x;\n\n";
    s += "    Acc acc = idty();\n";
    s += "    for ( unsigned long long i = (unsigned long long) blockIdx.x*blockDim.x + tid;\n";
    s += "          i < N; i += (unsigned long long) blockDim.x*gridDim.x )\n";
    s += "        acc = cmb( acc, "; s += loadCall; s += " );\n";
    s += "    sdata[tid] = acc;\n";
    s += "    __syncthreads();\n\n";
    s += "    for ( unsigned k = blockDim.x >> 1; k > 0; k >>= 1 ) {\n";
    s += "        if ( tid < k ) sdata[tid] = cmb( sdata[tid], sdata[tid+k] );\n";
    s += "        __syncthreads();\n";
    s += "    }\n\n";
    s += "    if ( tid == 0 ) {\n";
    s += "        partials[blockIdx.x] = sdata[0];\n";
    s += "        __threadfence();\n";
    s += "        unsigned int ticket = atomicAdd( "; s += counterAddr; s += ", 1u );\n";
    s += "        s_last = ( ticket == gridDim.x - 1 );\n";
    s += "    }\n";
    s += "    __syncthreads();\n\n";
    s += "    if ( s_last ) {\n";
    s += "        Acc total = idty();\n";
    s += "        for ( unsigned i = tid; i < gridDim.x; i += blockDim.x )\n";
    s += "            total = cmb( total, partials[i] );\n";
    s += "        sdata[tid] = total;\n";
    s += "        __syncthreads();\n";
    s += "        for ( unsigned k = blockDim.x >> 1; k > 0; k >>= 1 ) {\n";
    s += "            if ( tid < k ) sdata[tid] = cmb( sdata[tid], sdata[tid+k] );\n";
    s += "            __syncthreads();\n";
    s += "        }\n";
    s += "        if ( tid == 0 ) { *out = sdata[0]; "; s += counterReset; s += " }\n";
    s += "    }\n";
    s += "}\n";
    return s;
}

static void
printSignature( const std::string &src )
{
    size_t a = src.find("\nreduce( ");
    size_t b = src.find(" )\n{", a);
    if ( a != std::string::npos && b != std::string::npos )
        printf("   generated: reduce%s )\n", src.substr(a + 7, b - (a + 7)).c_str());
}

//
// Compile CUDA C++ source to PTX for the current device's architecture.
// Returns the PTX, or an empty string on failure.
//
static std::string
compilePTX( const std::string &src, int major, int minor )
{
    nvrtcResult status_nvrtc;
    nvrtcProgram prog = 0;
    std::string ptx;
    size_t ptxSize = 0, logSize = 0;
    char arch[64];
    snprintf( arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor );
    const char *opts[] = { arch };

    nvrtc( CreateProgram( &prog, src.c_str(), "reduce.cu", 0, NULL, NULL ) );
    status_nvrtc = nvrtcCompileProgram( prog, 1, opts );
    nvrtcGetProgramLogSize( prog, &logSize );
    if ( logSize > 1 ) {
        std::string log( logSize, '\0' );
        nvrtcGetProgramLog( prog, &log[0] );
        printf( "%s\n", log.c_str() );
    }
    if ( NVRTC_SUCCESS != status_nvrtc ) goto Error_nvrtc;

    nvrtc( GetPTXSize( prog, &ptxSize ) );
    ptx.resize( ptxSize );
    nvrtc( GetPTX( prog, &ptx[0] ) );
    nvrtc( DestroyProgram( &prog ) );
    return ptx;
Error_nvrtc:
    if ( prog ) nvrtcDestroyProgram( &prog );
    return std::string();
}

//
// Launch the generated kernel, building the argument list to match the
// interface NVRTC produced: [out, partials, (retire?), (ctx?), in, N]. Pass
// d_retire = 0 when the counter is a module global, d_ctx = 0 when there is
// no captured context. Returns 0 on success.
//
static int
launchReduce( CUfunction fn, int grid, int block, int sharedBytes,
              void *d_out, void *d_partials, unsigned int *d_retire,
              void *d_ctx, void *d_in, unsigned long long *pN )
{
    cudaError_t status_cudart;
    CUresult status_cuda;
    void *args[8]; int n = 0;
    args[n++] = &d_out;
    args[n++] = &d_partials;
    if ( d_retire ) args[n++] = &d_retire;
    if ( d_ctx )    args[n++] = &d_ctx;
    args[n++] = &d_in;
    args[n++] = pN;
    cu( LaunchKernel( fn, grid,1,1, block,1,1, sharedBytes, 0, args, 0 ) );
    cuda( DeviceSynchronize() );
    return 0;
Error_cudart:
    return (int) status_cudart;
Error_cuda:
    return (int) status_cuda;
}

// host-side views + thrust functors for the compile-time comparison
struct Stats { long long sum, sumsq; };
struct Ctx   { int thresh; };
struct ToStats   { __host__ __device__ Stats operator()( int x ) const { return Stats{ (long long)x, (long long)x*x }; } };
struct PlusStats { __host__ __device__ Stats operator()( const Stats &a, const Stats &b ) const { return Stats{ a.sum+b.sum, a.sumsq+b.sumsq }; } };
struct Above     { int t; __host__ __device__ bool operator()( int x ) const { return x > t; } };

int
main( int argc, char *argv[] )
{
    // Declare everything up front so the error-check macros can goto Error
    // without jumping over an initialization.
    cudaError_t status_cudart;
    CUresult status_cuda;
    size_t N   = ( argc > 1 ) ? (size_t) atoll(argv[1]) : (16u << 20);
    int    thr = ( argc > 2 ) ? atoi(argv[2]) : 0;
    const int block = 256, grid = 1024;
    CUdevice  dev = 0;
    CUcontext ctx = 0;
    int major = 0, minor = 0;
    int *h_in = 0;
    long long refSum = 0, refSumsq = 0, refCount = 0;
    int *d_in = 0;
    void *d_out = 0, *d_partials = 0;
    unsigned int *d_retire = 0;
    unsigned long long Nll = N;

    Monoid sumsq = {
        "int", "struct Acc { long long sum, sumsq; };", NULL,
        "Acc r; r.sum = x; r.sumsq = (long long) x * x; return r;",
        "Acc r; r.sum = a.sum + b.sum; r.sumsq = a.sumsq + b.sumsq; return r;",
        "Acc r; r.sum = 0; r.sumsq = 0; return r;"
    };
    Monoid count = {
        "int", "typedef long long Acc;", "struct Ctx { int thresh; };",
        "return x > c.thresh ? 1 : 0;",
        "return a + b;",
        "return 0;"
    };

    // Use the device's primary context so runtime (cudaMalloc) and driver
    // (module launch) allocations share one context.
    cu( Init(0) );
    cu( DeviceGet(&dev, 0) );
    cu( DevicePrimaryCtxRetain(&ctx, dev) );
    cu( CtxSetCurrent(ctx) );
    cu( DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) );
    cu( DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) );

    h_in = (int *) malloc( N*sizeof(int) );
    srand(1);
    for ( size_t i=0; i<N; i++ ) { int v=(rand()&7)-3; h_in[i]=v; refSum+=v; refSumsq+=(long long)v*v; if (v>thr) refCount++; }

    cuda( Malloc(&d_in, N*sizeof(int)) );
    cuda( Memcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice) );
    cuda( Malloc(&d_out, sizeof(Stats)) );                 // widest Acc we use
    cuda( Malloc(&d_partials, grid*sizeof(Stats)) );
    cuda( Malloc(&d_retire, sizeof(unsigned)) );

    printf( "N=%zu  sm_%d%d  block=%d grid=%d\n", N, major, minor, block, grid );
    printf( "CPU reference:  sum=%lld  sumsq=%lld  count(>%d)=%lld\n\n", refSum, refSumsq, thr, refCount );

    // ===== monoid 1: sum + sum of squares (struct accumulator, no context) =====
    printf( "== monoid 1: sum + sum of squares ==\n" );
    {
        std::string src = buildSource( sumsq, /*globalCounter=*/false );
        printSignature( src );
        std::string ptx = compilePTX( src, major, minor );
        if ( ptx.empty() ) goto Error_cudart;
        CUmodule mod; cu( ModuleLoadData(&mod, ptx.c_str()) );
        CUfunction fn; cu( ModuleGetFunction(&fn, mod, "reduce") );
        cuda( Memset(d_retire, 0, sizeof(unsigned)) );
        if ( launchReduce( fn, grid, block, block*sizeof(Stats), d_out, d_partials, d_retire, 0, d_in, &Nll ) ) goto Error_cudart;
        Stats o{}; cuda( Memcpy(&o, d_out, sizeof o, cudaMemcpyDeviceToHost) );
        printf( "   (B) counter as argument:      sum=%lld sumsq=%lld   %s\n", o.sum, o.sumsq, (o.sum==refSum&&o.sumsq==refSumsq)?"ok":"FAIL" );
        cu( ModuleUnload(mod) );
    }
    {
        std::string src = buildSource( sumsq, /*globalCounter=*/true );
        printSignature( src );
        std::string ptx = compilePTX( src, major, minor );
        if ( ptx.empty() ) goto Error_cudart;
        CUmodule mod; cu( ModuleLoadData(&mod, ptx.c_str()) );
        CUfunction fn; cu( ModuleGetFunction(&fn, mod, "reduce") );
        CUdeviceptr g=0; size_t gb=0; cu( ModuleGetGlobal(&g, &gb, mod, "retirementCount") );
        unsigned z=0; cu( MemcpyHtoD(g, &z, sizeof z) );          // reach the module global from the host
        Stats o{}; void *dout=d_out, *dpar=d_partials, *din=d_in;
        void *args[]={&dout,&dpar,&din,&Nll};
        cu( LaunchKernel(fn, grid,1,1, block,1,1, block*sizeof(Stats), 0, args, 0) );
        cuda( DeviceSynchronize() );
        cuda( Memcpy(&o, d_out, sizeof o, cudaMemcpyDeviceToHost) );
        printf( "   (A) counter as module global (%zuB via cuModuleGetGlobal): sum=%lld sumsq=%lld   %s\n", gb, o.sum, o.sumsq, (o.sum==refSum&&o.sumsq==refSumsq)?"ok":"FAIL" );
        cu( ModuleUnload(mod) );
    }
    {
        thrust::device_ptr<int> p(d_in);
        Stats o = thrust::transform_reduce( thrust::device, p, p+N, ToStats(), Stats{0,0}, PlusStats() );
        printf( "   thrust::transform_reduce:     sum=%lld sumsq=%lld   %s\n", o.sum, o.sumsq, (o.sum==refSum&&o.sumsq==refSumsq)?"ok":"FAIL" );
    }

    // ===== monoid 2: count elements > thresh (scalar Acc, captured context) =====
    printf( "\n== monoid 2: count elements > thresh=%d (captured context) ==\n", thr );
    {
        std::string src = buildSource( count, /*globalCounter=*/false );
        printSignature( src );                                        // note the extra const Ctx* parameter
        std::string ptx = compilePTX( src, major, minor );
        if ( ptx.empty() ) goto Error_cudart;
        CUmodule mod; cu( ModuleLoadData(&mod, ptx.c_str()) );
        CUfunction fn; cu( ModuleGetFunction(&fn, mod, "reduce") );
        Ctx *d_ctx=0; cuda( Malloc(&d_ctx, sizeof(Ctx)) );
        Ctx hc{ thr }; cuda( Memcpy(d_ctx, &hc, sizeof hc, cudaMemcpyHostToDevice) );  // the "capture"
        cuda( Memset(d_retire, 0, sizeof(unsigned)) );
        if ( launchReduce( fn, grid, block, block*sizeof(long long), d_out, d_partials, d_retire, d_ctx, d_in, &Nll ) ) goto Error_cudart;
        long long o=0; cuda( Memcpy(&o, d_out, sizeof o, cudaMemcpyDeviceToHost) );
        printf( "   NVRTC (context passed as arg):        count=%lld   %s\n", o, (o==refCount)?"ok":"FAIL" );
        cudaFree(d_ctx); cu( ModuleUnload(mod) );
    }
    {
        thrust::device_ptr<int> p(d_in);
        long long o = thrust::count_if( thrust::device, p, p+N, Above{ thr } );  // runtime value via a stateful functor
        printf( "   thrust::count_if (stateful functor):  count=%lld   %s\n", o, (o==refCount)?"ok":"FAIL" );
    }

    free(h_in); cudaFree(d_in); cudaFree(d_out); cudaFree(d_partials); cudaFree(d_retire);
    return 0;
Error_cudart:
Error_cuda:
    free(h_in); cudaFree(d_in); cudaFree(d_out); cudaFree(d_partials); cudaFree(d_retire);
    return 1;
}
