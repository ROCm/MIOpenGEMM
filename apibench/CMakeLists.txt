################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################


SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

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
