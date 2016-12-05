SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic " )

include(GNUInstallDirs)

configure_file(config.hpp.dev dev_include/tinygemm/config.hpp)
configure_file(config.hpp.install install_include/config.hpp)

file(GLOB_RECURSE source_files src/*.cpp)

add_library(tinygemm ${source_files})

target_link_libraries(tinygemm ${OPENCL_LIBRARIES})

target_include_directories (tinygemm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>
    ${OPENCL_INCLUDE_DIRS})
