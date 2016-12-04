SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic " )

add_definitions(-DPATH_MAKE_KERNEL_CMDL_PY="${CMAKE_CURRENT_SOURCE_DIR}/python/make_kernel_cmdl.py")

add_definitions(-DPATH_BETAC_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/opencl/")

file(GLOB_RECURSE source_files src/*.cpp)

add_library(tinygemm ${source_files})

target_link_libraries(tinygemm ${OPENCL_LIBRARIES})

target_include_directories (tinygemm PUBLIC include ${OPENCL_INCLUDE_DIRS})
