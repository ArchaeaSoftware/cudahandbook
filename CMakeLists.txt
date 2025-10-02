cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Define the project and enable CUDA and C++ languages
project(cudahandbook LANGUAGES CUDA CXX)

# Set the C++ standard for C++ files (optional, but good practice)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA standard for CUDA files (optional, but good practice)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional: Enable separable compilation for better performance and modularity
# This allows kernels and device functions to be compiled separately and linked later.
#set_target_properties(MyCUDALibrary PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add an executable target

#add_executable(my_cuda_app
#    main.cpp
#    kernel
#    device_functions
#)

add_custom_target( cudahandbook )


# Optional: Add an include directory if you have header files in a specific location
include_directories( chLib )

add_subdirectory( chLib )
add_subdirectory( histogram )
add_subdirectory( streaming )
add_subdirectory( microbench )
add_subdirectory( concurrency )
add_subdirectory( memory )
add_subdirectory( reduction )
add_subdirectory( corr )
add_subdirectory( SMs )
add_subdirectory( scan )
add_subdirectory( texturing )
add_subdirectory( nbody )

# Optional: Link to a CUDA library (e.g., if you have a separate CUDA library)
#target_link_libraries(cudahandbook chLib)

# Optional: Set CUDA architecture for compilation (e.g., for specific GPU models)

set_target_properties( chLib

breakevenDtoHMemcpy             nullHtoDMemcpyAsync
breakevenHtoDMemcpy             nullKernelAsync
breakevenKernelAsync            nullKernelAsyncGraph
concurrencyKernelKernel         nullKernelSync
concurrencyKernelMapped         pageableMemcpyHtoD16Blocking
concurrencyMemcpyKernel         pageableMemcpyHtoD16Broken
concurrencyMemcpyKernelMapped   pageableMemcpyHtoD16
cudaGetLastErrorIsAsynchronous  pageableMemcpyHtoD16Synchronous
eventRecord                     pageableMemcpyHtoD
managedOverhead                 peer2peerMemcpy
nullDtoHMemcpyAsync
nullDtoHMemcpySync

AOStoSOA     globalRead     linuxGPULoad        peer2peerTestNUMA
globalCopy2  globalReadTex  peer2peerMemcpy     pinnedBandwidth
globalCopy   globalWrite    peer2peerMemcpyDrv  spinlockReduction
	               
	               histogram
                       normalizedCrossCorrelation

                       globalRead globalWrite mallocSpeed reportClocks
		       reduction
	               stream1Device stream2Async stream3Streams stream4Mapped stream5MappedXfer

		       divergence float_to_float16 solveQuadratic testFunnelShift testShuffle

		       #testScanBlock testScanWarp testStreamCompact_odd timeStreamCompact_odd
		       testScan timeScan

		       surf1Dmemset tex1d_addressing tex1dfetch_htod tex2d_addressing tex2d_shmoo
		       surf2Dmemset tex1dfetch_big tex1dfetch_int2float tex2d_addressing_device
		       tex1dfetch_offset surf2Dmemset_shmoo tex1dfetch tex2d_memset tex1d_9bit
		       tex1dfetch_host tex1d_unnormalized tex2d_opengl

		       nbody

	               PROPERTIES CUDA_ARCHITECTURES 80-virtual )
if (WIN32)
	set_target_properties( win32GPULoad win32NullKernelStreamCallback PROPERTIES CUDA_ARCHITECTURES 80-virtual )
endif()


