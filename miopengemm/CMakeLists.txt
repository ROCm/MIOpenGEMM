################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

add_definitions(-std=c++11)

set( miopengemm_INSTALL_DIR miopengemm)
set( lib_SOVERSION 1.0 )

option(BUILD_DEV OFF)

configure_file(config.hpp.install dev_include/miopengemm/config.hpp)

file(GLOB_RECURSE source_files src/*.cpp)

add_library(miopengemm ${source_files})

target_link_libraries(miopengemm PUBLIC ${OPENCL_LIBRARIES} ${OpenBLAS_LIB} ${CLBLAST_LIB} ${ISAAC_LIB})

if(NOT WIN32 AND NOT APPLE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lib.def "

      MIOPENGEMM_1
      {
      global:
          miopengemm*;
          *MIOpenGEMM*;
          extern \"C++\" {
              MIOpenGEMM::*;
          };
      local:
          *;
      };
    ")

    target_link_libraries(miopengemm PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/lib.def")
    target_link_libraries(miopengemm PRIVATE "-Wl,--exclude-libs,ALL")
    rocm_set_soversion(miopengemm ${lib_SOVERSION})
    set_target_properties(miopengemm PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()

target_include_directories (miopengemm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>)
target_include_directories (miopengemm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
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
)

install(
  DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/include/miopengemm/"
  "${CMAKE_CURRENT_BINARY_DIR}/dev_include/miopengemm/"
  DESTINATION "include/miopengemm/"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

rocm_export_targets(
  TARGETS miopengemm
)

rocm_create_package(
    NAME miopengemm
    DESCRIPTION "A tool for generating OpenCL matrix multiplication (GEMM) kernels"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
)

