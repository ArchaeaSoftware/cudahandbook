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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gemm LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# <chError.h> and friends live in chLib/. The top-level build adds this
# globally; name it here too so the directory also builds standalone.
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../chLib )

# The SGEMM optimization journey of Chapter 17, one executable per stage.
# Stages 5 and 6 use TF32 Tensor Cores and cp.async, which require sm_80+.
add_executable( sgemm1Naive     sgemm1Naive.cu )
add_executable( sgemm2Cache     sgemm2Cache.cu )
add_executable( sgemm3Tiled     sgemm3Tiled.cu )
add_executable( sgemm4RegBlock  sgemm4RegBlock.cu )
add_executable( sgemm5Wmma      sgemm5Wmma.cu )
add_executable( sgemm6WmmaAsync sgemm6WmmaAsync.cu )

foreach( t sgemm1Naive sgemm2Cache sgemm3Tiled sgemm4RegBlock sgemm5Wmma sgemm6WmmaAsync )
    set_target_properties( ${t} PROPERTIES CUDA_ARCHITECTURES "80" )
    target_link_libraries( ${t} cublas )
endforeach()
