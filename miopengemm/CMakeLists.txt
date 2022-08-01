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
# Opencl headers installed in cmake install includedir and INTERFACE_INCLUDE_DIRECTORIES have that path
# OPENCL_INCLUDE_DIRS is redundant and not required in install interface.
# Required for building, so limiting it to build interface
target_include_directories (miopengemm SYSTEM PUBLIC

    $<BUILD_INTERFACE:${OPENCL_INCLUDE_DIRS}>
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
)

rocm_export_targets(
  TARGETS miopengemm
)

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/miopengemm/include"
        DESTINATION "miopengemm" )
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()


rocm_create_package(
    NAME miopengemm
    DESCRIPTION "A tool for generating OpenCL matrix multiplication (GEMM) kernels"
    MAINTAINER "MIOpen Maintainer <miopen-lib.support@amd.com>"
    LDCONFIG
)

