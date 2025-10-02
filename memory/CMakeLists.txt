cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Define the project and enable CUDA and C++ languages
project(memory LANGUAGES CUDA CXX)

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

add_executable( AOStoSOA AOStoSOA.cu )
add_executable( globalRead globalRead.cu )
add_executable( linuxGPULoad linuxGPULoad.cu)
add_executable( peer2peerTestNUMA peer2peerTestNUMA.cu )
if (WIN32)
    add_executable( win32GPULoad win32GPULoad.cu )
endif()
add_executable( globalCopy2 globalCopy2.cu )
add_executable( globalReadTex globalReadTex.cu )
add_executable( peer2peerMemcpy peer2peerMemcpy.cu )
add_executable( pinnedBandwidth pinnedBandwidth.cu )
add_executable( globalCopy globalCopy.cu )
add_executable( globalWrite globalWrite.cu )
add_executable( peer2peerMemcpyDrv peer2peerMemcpyDrv.cu )
add_executable( spinlockReduction spinlockReduction.cu )

target_link_libraries( linuxGPULoad numa )
target_link_libraries( peer2peerTestNUMA numa )
target_link_libraries( peer2peerMemcpyDrv cuda )

# Optional: Add an include directory if you have header files in a specific location
include_directories( chLib )

