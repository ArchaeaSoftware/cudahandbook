cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Define the project and enable CUDA and C++ languages
project(microbench LANGUAGES CUDA CXX)

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

add_executable( reportClocks reportClocks.cu )
#add_executable( globalRead globalRead.cu )
#add_executable( globalWrite globalWrite.cu )
add_executable( mallocSpeed mallocSpeed.cu )

# Optional: Add an include directory if you have header files in a specific location
include_directories( chLib )

# Optional: Link to a CUDA library (e.g., if you have a separate CUDA library)
target_link_libraries(reportClocks chLib)

