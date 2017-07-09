################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

add_custom_target(examples COMMENT "Build all examples.")

function(add_example_executable EX_NAME)
    add_executable (${EX_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${EX_NAME} miopengemm)
    add_dependencies(examples ${EX_NAME})
endfunction(add_example_executable)

add_example_executable(basicexample basicexample.cpp)
add_example_executable(deepbench deepbench.cpp)
add_example_executable(redirectionexample redirectionexample.cpp)
add_example_executable(devtest devtest.cpp)
add_example_executable(experiment1 experiment1.cpp)
add_example_executable(initialisationdemo initialisationdemo.cpp)
add_example_executable(gencache gencache.cpp)
add_example_executable(benchthecache benchthecache.cpp)
