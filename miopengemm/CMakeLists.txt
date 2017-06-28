################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
################################################################################
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic " )

set( miopengemm_INSTALL_DIR miopengemm)

option(BUILD_DEV OFF)

if(BUILD_DEV)
configure_file(config.hpp.dev dev_include/miopengemm/config.hpp)
else()
configure_file(config.hpp.install dev_include/miopengemm/config.hpp)
endif()

file(GLOB_RECURSE source_files src/*.cpp)

add_library(miopengemm ${source_files})

target_link_libraries(miopengemm ${OPENCL_LIBRARIES})

target_include_directories (miopengemm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>
    ${OPENCL_INCLUDE_DIRS})

rocm_install_targets(
  TARGETS miopengemm
  INCLUDE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PREFIX ${miopengemm_INSTALL_DIR}
)

rocm_export_targets(
  TARGETS miopengemm
  PREFIX ${miopengemm_INSTALL_DIR}
)

rocm_create_package(
    NAME miopengemm
    DESCRIPTION "A tool for generating OpenCL matrix multiplication (GEMM) kernels"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
)

rocm_install_symlink_subdir(${miopengemm_INSTALL_DIR})

