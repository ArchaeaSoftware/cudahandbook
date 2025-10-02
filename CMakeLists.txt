#
# Copyright (c) 2025, Archaea Software, LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cudahandbook LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_custom_target( cudahandbook )

include_directories( chLib )

add_subdirectory( chLib )
add_subdirectory( corr )
add_subdirectory( concurrency )
add_subdirectory( histogram )
add_subdirectory( memory )
add_subdirectory( microbench )
add_subdirectory( nbody )
add_subdirectory( reduction )
add_subdirectory( scan )
add_subdirectory( SMs )
add_subdirectory( streaming )
add_subdirectory( texturing )

# Set CUDA architecture for compilation (e.g., for specific GPU models)

set_target_properties(
    chLib

    # concurrency
    breakevenDtoHMemcpy
    breakevenHtoDMemcpy
    breakevenKernelAsync
    concurrencyKernelKernel
    concurrencyKernelMapped
    concurrencyMemcpyKernel
    concurrencyMemcpyKernelMapped
    cudaGetLastErrorIsAsynchronous
    eventRecord
    managedOverhead
    nullDtoHMemcpyAsync
    nullDtoHMemcpySync
    nullHtoDMemcpyAsync
    nullKernelAsync
    nullKernelAsyncGraph
    nullKernelSync
    pageableMemcpyHtoD16Blocking
    pageableMemcpyHtoD16Broken
    pageableMemcpyHtoD16
    pageableMemcpyHtoD16Synchronous
    pageableMemcpyHtoD
    peer2peerMemcpy

    # memory
    AOStoSOA
    globalCopy
    globalCopy2
    globalRead
    globalReadTex
    globalWrite
    peer2peerTestNUMA
    peer2peerMemcpy
    peer2peerMemcpyDrv
    pinnedBandwidth
    spinlockReduction

    # histogram
    histogram

    # corr
    normalizedCrossCorrelation

    # microbench
    globalRead
    globalWrite
    mallocSpeed
    reportClocks

    # reduction
    reduction

    # streaming
    stream1Device stream2Async stream3Streams stream4Mapped stream5MappedXfer

    # SMs
    divergence float_to_float16 solveQuadratic testFunnelShift testShuffle

    # scan
    #testScanBlock testScanWarp testStreamCompact_odd timeStreamCompact_odd
    testScan timeScan

    # texturing
    surf1Dmemset
    surf2Dmemset
    surf2Dmemset_shmoo
    tex1d_addressing
    tex1dfetch
    tex1d_9bit
    tex1dfetch_big
    tex1dfetch_htod
    tex1dfetch_host
    tex1dfetch_int2float
    tex1dfetch_offset
    tex1d_unnormalized
    tex2d_addressing
    tex2d_shmoo
    tex2d_addressing_device
    tex2d_memset
    tex2d_opengl

    nbody

	               PROPERTIES CUDA_ARCHITECTURES 80-virtual )
if (WIN32)
    set_target_properties( win32GPULoad win32NullKernelStreamCallback PROPERTIES CUDA_ARCHITECTURES 80-virtual )
endif()

if ( UNIX )
    set_target_properties( linuxGPULoad PROPERTIES CUDA_ARCHITECTURES 80-virtual )
endif()

