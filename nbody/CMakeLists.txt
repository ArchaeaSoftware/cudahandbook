cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Define the project and enable CUDA and C++ languages
project(nbody LANGUAGES CUDA CXX)

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

add_executable(nbody nbody.cu nbody_CPU_SSE.cpp nbody_CPU_AOS.cpp nbody_CPU_AOS_tiled.cpp nbody_CPU_SSE_threaded.cpp nbody_CPU_SOA.cpp nbody_GPU_shared.cu nbody_multiGPU.cu nbody_multiGPU_threaded.cu )

# Optional: Add an include directory if you have header files in a specific location
include_directories( chLib )

