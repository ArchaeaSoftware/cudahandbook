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

project(memory LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

include_directories( chLib )

