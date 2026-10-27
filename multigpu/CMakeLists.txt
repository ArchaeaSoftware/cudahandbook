#
# Copyright (c) 2025-2026, Archaea Software, LLC.
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

project(multigpu LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# NCCL is not part of the CUDA toolkit, so look for it where it is usually
# found: a prefix the user names, one unpacked into a home directory on a
# machine without root, or the system paths a Deep Learning AMI uses.
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${NCCL_HOME} $ENV{NCCL_HOME} $ENV{HOME}/opt/nccl /usr/local/nccl
    PATH_SUFFIXES include)
find_library(NCCL_LIBRARY nccl
    HINTS ${NCCL_HOME} $ENV{NCCL_HOME} $ENV{HOME}/opt/nccl /usr/local/nccl
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "NCCL: ${NCCL_LIBRARY}")
    add_executable(ncclProbe ncclProbe.cu)
    target_include_directories(ncclProbe PRIVATE chLib ${NCCL_INCLUDE_DIR})
    target_link_libraries(ncclProbe PRIVATE ${NCCL_LIBRARY})
    # Record the library's location in the binary, so a prefix outside the
    # loader's search path does not have to be named again at run time.
    get_filename_component(NCCL_LIBRARY_DIR ${NCCL_LIBRARY} DIRECTORY)
    set_target_properties(ncclProbe PROPERTIES
        BUILD_RPATH ${NCCL_LIBRARY_DIR}
        INSTALL_RPATH ${NCCL_LIBRARY_DIR})
else()
    message(STATUS "NCCL not found; skipping the multigpu targets. "
                   "Set -DNCCL_HOME=<prefix> to build them.")
endif()

include_directories( chLib )
