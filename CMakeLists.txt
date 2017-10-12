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

find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)

rocm_setup_version(VERSION 1.1.5)

#Where to find FindOpenCL.cmake and other files which help locate external files and libraries
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(OPENBLAS "CPU GEMM accuracy tests with OpenBLAS, otherwise slow 3-for loops code" OFF)
option(API_BENCH_MIOGEMM "Build benchmarking of MIOpenGEMM" OFF)
option(API_BENCH_CLBLAST "Build benchmarking of CLBlast" OFF)
option(API_BENCH_ISAAC "Build benchmarking of Isaac" OFF)

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
