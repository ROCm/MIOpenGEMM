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

add_example_executable(accu accu.cpp)
add_example_executable(find find.cpp)
add_example_executable(bench bench.cpp)
add_example_executable(gencache gencache.cpp)
add_example_executable(redirection redirection.cpp)
add_example_executable(runcache runcache.cpp)
add_example_executable(initialisationdemo initialisationdemo.cpp)
add_example_executable(multifind multifind.cpp)
add_example_executable(multifindbase multifindbase.cpp)
add_example_executable(mergecaches mergecaches.cpp)
add_example_executable(genrocmtest genrocmtest.cpp)
add_example_executable(apiexample1 apiexample1.cpp)
add_example_executable(apidriver apidriver.cpp)
add_example_executable(deepbench deepbench.cpp)
add_example_executable(gemmbench gemmbench.cpp)
add_example_executable(print print.cpp)
