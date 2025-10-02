cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Define the project and enable CUDA and C++ languages
project(texturing LANGUAGES CUDA CXX)

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

add_executable( surf1Dmemset surf1Dmemset.cu )
add_executable( tex1d_addressing tex1d_addressing.cu )
add_executable( tex1dfetch_htod tex1dfetch_htod.cu )
add_executable( tex2d_addressing tex2d_addressing.cu )
add_executable( tex2d_shmoo tex2d_shmoo.cu )
add_executable( surf2Dmemset surf2Dmemset.cu )
add_executable( tex1dfetch_big tex1dfetch_big.cu )
add_executable( tex1dfetch_int2float tex1dfetch_int2float.cu )
add_executable( tex2d_addressing_device tex2d_addressing_device.cu )
add_executable( tex1dfetch_offset tex1dfetch_offset.cu )
add_executable( surf2Dmemset_shmoo surf2Dmemset_shmoo.cu )
add_executable( tex1dfetch tex1dfetch.cu )
add_executable( tex2d_memset tex2d_memset.cu )
add_executable( tex1d_9bit tex1d_9bit.cu )
add_executable( tex1dfetch_host tex1dfetch_host.cu )
add_executable( tex1d_unnormalized tex1d_unnormalized.cu )
add_executable( tex2d_opengl tex2d_opengl.cu )

# Optional: Add an include directory if you have header files in a specific location
include_directories( chLib )

target_link_libraries( tex2d_shmoo cuda )
target_link_libraries( surf2Dmemset cuda )
target_link_libraries( surf2Dmemset_shmoo cuda )
target_link_libraries( tex1d_9bit cuda )
target_link_libraries( tex2d_opengl GL GLU glut )

