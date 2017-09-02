################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

#cmake_minimum_required(VERSION 3.0)

#project(miopengemm)

#option(OPENBLAS "CPU GEMM accuracy tests with OpenBLAS, otherwise slow 3-for loops code" OFF)

##Where to find FindOpenCL.cmake (etc)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" )

#find_package(MIOpenGEMM REQUIRED)
#find_package(OpenCL REQUIRED)
#find_package(OpenBLAS REQUIRED)
#find_package(CLBlast REQUIRED)
#find_package(Isaac REQUIRED)

#option(BUILD_SHARED_LIBS "Build shared library" ON)

#include(EnableCompilerWarnings)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

#add_custom_target(examples COMMENT "Build all examples.")

function(add_apibench_executable EX_NAME)
    add_executable (${EX_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${EX_NAME} ${OPENCL_LIBRARIES} ${OpenBLAS_LIB} ${CLBLAST_LIB} ${ISAAC_LIB} miopengemm)
    target_include_directories ( ${EX_NAME} PUBLIC 
    ${OPENCL_INCLUDE_DIRS} 
    ${CLBLAST_INCLUDE_DIR}
    ${OpenBLAS_INCLUDE_DIR}
    ${MIOPENGEMM_INCLUDE_DIR}
)

#add_dependencies(examples ${EX_NAME})
endfunction(add_apibench_executable)

add_apibench_executable(gemmbench gemmbench.cpp)
