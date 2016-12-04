SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

# include_directories("../tinygemm/include" ${OpenCL_INCLUDE_DIR})

include(CTest)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests COMMENT "Build all tests.")
add_dependencies(check tests)

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME} tinygemm)
    if(WIN32)
        add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    else()
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    endif()
    add_dependencies(tests ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
endfunction(add_test_executable)

add_test_executable(basicexample basicexample.cpp)

add_test_executable(deepbench deepbench.cpp)

add_test_executable(backconvwrw backconvwrw.cpp)

add_test_executable(smallgeometrytests smallgeometrytests.cpp)

add_test_executable(redirectionexample redirectionexample.cpp)

add_test_executable(devtest devtest.cpp)
