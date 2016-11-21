SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )

include_directories("../tinygemm/include" ${OpenCL_INCLUDE_DIR})
#Déclaration of executable
add_executable(
tinygemmtest tinygemmtest.cpp)

target_link_libraries(
tinygemmtest LINK_PUBLIC tinygemm ${OpenCL_LIBRARY})
