#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

__device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
{
    uint result;
    asm(
        "{.reg .u32 r0;"
         ".reg .pred p;"
         "shfl.up.b32 r0|p, %1, %2, 0;"
         "@p add.u32 r0, r0, %3;"
         "mov.u32 %0, r0;}"
        : "=r"(result) : "r"(partial), "r"(up_offset), "r"(partial));
    return result;
}

template <int levels>
__device__ __forceinline__ uint inclusive_scan_warp_shfl(int mysum)
{
    // this pragma may be unnecessary with the template parameter!
    #pragma unroll
    for(int i = 0; i < levels; ++i)
        mysum = shfl_scan_add_step(mysum, 1 << i);
    return mysum;
}

template <int logBlockSize>
__device__ uint inclusive_scan_block(uint val, const unsigned int idx)
{
    const unsigned int lane   = idx & 31;
    const unsigned int warpid = idx >> 5;
    __shared__ uint ptr[WARP_SIZE];

    // step 1: Intra-warp scan in each warp

    val = inclusive_scan_warp_shfl<LOG_WARP_SIZE>(val);

    // step 2: Collect per-warp particle results
    if (lane == 31) ptr[warpid] = val;
    __syncthreads();
    // step 3: Use 1st warp to scan per-warp results
    if (warpid == 0) ptr[lane] = inclusive_scan_warp_shfl<logBlockSize-LOG_WARP_SIZE>(ptr[lane]);
    __syncthreads();
    // step 4: Accumulate results from Steps 1 and 3;
    if (warpid > 0) val += ptr[warpid - 1];
    // __syncthreads(); // MJH don't think this sync is needed since we have a function-scope
    // shared memory array. But you might want to use a shared allocation that gets reused later
    // to cut down shared  usage even further.  In that case either sync there or before you
    // write to the aliased shared memory addresses upon returning from this call.
    return val;
}
