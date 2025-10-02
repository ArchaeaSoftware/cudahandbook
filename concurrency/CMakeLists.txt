cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Define the project and enable CUDA and C++ languages
project(concurrency LANGUAGES CUDA CXX)

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

add_executable( breakevenDtoHMemcpy breakevenDtoHMemcpy.cu )
add_executable( breakevenHtoDMemcpy breakevenHtoDMemcpy.cu )
add_executable( breakevenKernelAsync breakevenKernelAsync.cu )

add_executable( concurrencyKernelKernel concurrencyKernelKernel.cu )
add_executable( concurrencyKernelMapped concurrencyKernelMapped.cu )
add_executable( concurrencyMemcpyKernel concurrencyMemcpyKernel.cu )
add_executable( concurrencyMemcpyKernelMapped concurrencyMemcpyKernelMapped.cu )

add_executable( cudaGetLastErrorIsAsynchronous cudaGetLastErrorIsAsynchronous.cu )

add_executable( eventRecord eventRecord.cu )

add_executable( managedOverhead managedOverhead.cu )

add_executable( nullDtoHMemcpyAsync nullDtoHMemcpyAsync.cu )
add_executable( nullDtoHMemcpySync nullDtoHMemcpySync.cu )
add_executable( nullHtoDMemcpyAsync nullHtoDMemcpyAsync.cu )

add_executable( nullKernelAsync nullKernelAsync.cu )
add_executable( nullKernelAsyncGraph nullKernelAsyncGraph.cu )
add_executable( nullKernelSync nullKernelSync.cu )

add_executable( pageableMemcpyHtoD pageableMemcpyHtoD.cu )
add_executable( pageableMemcpyHtoD16Blocking pageableMemcpyHtoD16Blocking.cu memcpy16.cpp )
add_executable( pageableMemcpyHtoD16Broken pageableMemcpyHtoD16Broken.cu memcpy16.cpp )
add_executable( pageableMemcpyHtoD16 pageableMemcpyHtoD16.cu memcpy16.cpp )
add_executable( pageableMemcpyHtoD16Synchronous pageableMemcpyHtoD16Synchronous.cu memcpy16.cpp )

#add_executable( peer2peerMemcpy peer2peerMemcpy.cu )

if (WIN32)
  add_executable( win32NullKernelStreamCallback win32NullKernelStreamCallback.cu )
endif()


# Optional: Add an include directory if you have header files in a specific location
include_directories( chLib )

# Optional: Link to a CUDA library (e.g., if you have a separate CUDA library)
#target_link_libraries(concurrency chLib)

# Optional: Set CUDA architecture for compilation (e.g., for specific GPU models)
#set_target_properties(histogram PROPERTIES CUDA_ARCHITECTURES 75-real CUDA_ARCHITECTURES 75-virtual)

