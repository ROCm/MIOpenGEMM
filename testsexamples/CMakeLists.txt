SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

include_directories("../tinygemm/include" ${OpenCL_INCLUDE_DIR})

include(CTest)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests COMMENT "Build all tests.")
add_dependencies(check tests)

add_executable(
devtest EXCLUDE_FROM_ALL devtest.cpp)

target_link_libraries(
devtest LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})

add_test(NAME devtest WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND $<TARGET_FILE:devtest>)
add_dependencies(tests devtest)


add_executable(
basicexample EXCLUDE_FROM_ALL basicexample.cpp)

add_executable(
deepbench EXCLUDE_FROM_ALL deepbench.cpp)

add_executable(
backconvwrw EXCLUDE_FROM_ALL backconvwrw.cpp)

add_executable(
smallgeometrytests EXCLUDE_FROM_ALL smallgeometrytests.cpp)

add_executable(
redirectionexample EXCLUDE_FROM_ALL redirectionexample.cpp)





target_link_libraries(
basicexample LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})

target_link_libraries(
deepbench LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})

target_link_libraries(
backconvwrw LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})

target_link_libraries(
redirectionexample LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})

target_link_libraries(
smallgeometrytests LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})
