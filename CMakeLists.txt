
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

rocm_setup_version(VERSION 0.1)

#Where to find FindOpenCL.cmake (etc)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
##Use FindOpenCL.cmake to find library and header file
find_package( OpenCL REQUIRED )

find_package(PythonInterp REQUIRED)

# What I would like here : user sets the above directory, and at configuration time the file system is generated.
# Currently, I assume that the tree structure of the writing dir is already setup. 
# add_definitions(-DDIR_FOR_WRITING="/home/james/miopengemmout")

option(BUILD_SHARED_LIBS "Build shared library" ON)

include(EnableCompilerWarnings)

add_subdirectory(miopengemm)
add_subdirectory(tests)

# development code. It requires cython (and only builds with certain versions. You probably want to comment this out.
#add_subdirectory(dev)
