cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Define the project and enable CUDA and C++ languages
project(histogram LANGUAGES CUDA CXX)

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

add_executable( stream1Device stream1Device.cu )
add_executable( stream2Async stream2Async.cu )
add_executable( stream3Streams stream3Streams.cu )
add_executable( stream4Mapped stream4Mapped.cu )
add_executable( stream5MappedXfer stream5MappedXfer.cu )

# Optional: Add an include directory if you have header files in a specific location
include_directories( chLib )

# Optional: Link to a CUDA library (e.g., if you have a separate CUDA library)
target_link_libraries(chLib)

# Optional: Set CUDA architecture for compilation (e.g., for specific GPU models)
# set_target_properties(my_cuda_app PROPERTIES CUDA_ARCHITECTURES 75-real 75-virtual)

