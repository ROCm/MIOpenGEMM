################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

cmake_minimum_required(VERSION 3.0)

project(miopengemm)


#Where to find FindOpenCL.cmake (etc)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
##Use FindOpenCL.cmake to find library and header file
find_package( OpenCL REQUIRED )


option(BUILD_SHARED_LIBS "Build shared library" ON)

include(EnableCompilerWarnings)

include_directories(./miopengemm/include)
add_subdirectory(miopengemm)
add_subdirectory(examples)
add_subdirectory(tests)
