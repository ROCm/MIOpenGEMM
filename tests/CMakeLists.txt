################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

include(CTest)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests COMMENT "Build all tests.")
add_dependencies(check tests)

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME} miopengemm)
 
    #MIOpen has a conditional if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") on setting pthread. 
    set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
 
    if(WIN32)
        add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    else()
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    endif()
    add_dependencies(tests ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
endfunction(add_test_executable)


add_test_executable(smallgeometrytests smallgeometrytests.cpp)

add_test_executable(test_gemm0 test_gemm0.cpp)
