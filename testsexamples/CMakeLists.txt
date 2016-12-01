SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

include_directories("../tinygemm/include" ${OpenCL_INCLUDE_DIR})

add_executable(
basicexample basicexample.cpp)

add_executable(
deepbench deepbench.cpp)

add_executable(
backconvwrw backconvwrw.cpp)

add_executable(
smallgeometrytests smallgeometrytests.cpp)

add_executable(
redirectionexample redirectionexample.cpp)

add_executable(
devtest devtest.cpp)



target_link_libraries(
devtest LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})

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
