################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

add_definitions(-std=c++11)

set( miopengemm_INSTALL_DIR miopengemm)

option(BUILD_DEV OFF)

configure_file(config.hpp.install dev_include/miopengemm/config.hpp)

file(GLOB_RECURSE source_files src/*.cpp)

add_library(miopengemm ${source_files})

target_link_libraries(miopengemm ${OPENCL_LIBRARIES} ${OpenBLAS_LIB} ${CLBLAST_LIB} ${ISAAC_LIB})

target_include_directories (miopengemm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>)
target_include_directories (miopengemm SYSTEM PUBLIC 

    ${OPENCL_INCLUDE_DIRS} 
    ${CLBLAST_INCLUDE_DIR}
    ${ISAAC_INCLUDE_DIR}
    ${OpenBLAS_INCLUDE_DIR}
    )

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_include_directories (miopengemm PUBLIC ${OPENCL_INCLUDE_DIRS}/Headers)
endif()

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

