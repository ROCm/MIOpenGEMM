################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

cmake_minimum_required(VERSION 3.0)

# Default installation path
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/x86_64-w64-mingw32" CACHE PATH "Installation path")
else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Installation path")
endif()

project(miopengemm)

#find_package(ROCM REQUIRED PATHS /opt/rocm)
#Update Required ROCM-CMAKE Path for wrapper support
find_package(ROCM 0.7.3 CONFIG QUIET PATHS /opt/rocm ${ROCM_PATH})
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  set(rocm_cmake_url "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip")
  set(rocm_cmake_path "${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}")
  set(rocm_cmake_archive "${rocm_cmake_path}.zip")
  file(DOWNLOAD "${rocm_cmake_url}" "${rocm_cmake_archive}" STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(status_code EQUAL 0)
    message(STATUS "downloading... done")
  else()
    message(FATAL_ERROR "error: downloading\n'${rocm_cmake_url}' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}\n")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzvf "${rocm_cmake_archive}"
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(ROCMHeaderWrapper)

rocm_setup_version(VERSION 1.1.6)

#Where to find FindOpenCL.cmake and other files which help locate external files and libraries
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(OPENBLAS "CPU GEMM accuracy tests with OpenBLAS, otherwise slow 3-for loops code" OFF)
option(API_BENCH_MIOGEMM "Build benchmarking of MIOpenGEMM" OFF)
option(API_BENCH_CLBLAST "Build benchmarking of CLBlast" OFF)
option(API_BENCH_ISAAC "Build benchmarking of Isaac" OFF)

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" ON)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/miopengemm/include/miopengemm
    PATTERNS "*.hpp"
    GUARDS WRAPPER
    WRAPPER_LOCATIONS miopengemm/include/miopengemm
  )
endif()


if(OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    add_definitions(-DMIOPENGEMM_USE_OPENBLAS)
endif()

if (API_BENCH_CLBLAST)
    find_package(CLBlast REQUIRED)
    message("-- Adding definition MIOPENGEMM_BENCH_CLBLAST")
    add_definitions(-DMIOPENGEMM_BENCH_CLBLAST)
endif()

if (API_BENCH_ISAAC)
    find_package(Isaac REQUIRED)
    add_definitions(-DMIOPENGEMM_BENCH_ISAAC)
endif()


set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
find_package( OpenCL REQUIRED )

#SET (CMAKE_CXX_COMPILER "clang++")

include(EnableCompilerWarnings)
add_subdirectory(miopengemm)
add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(doc)
