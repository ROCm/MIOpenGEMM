cmake_minimum_required(VERSION 3.0)
project(TINYGEMM)

find_package( OpenCL REQUIRED )

# What I want to do here : uses sets the above directory, and at configuration time the file system is generated.
#Currently, I assume that the tree structure of the writing dir is already setup. 
add_definitions(-DDIR_FOR_WRITING="/home/james/somewhereoutthere")


#SET(CMAKE_CXX_COMPILER clang++)

add_subdirectory(tinygemm)
add_subdirectory(tests)
add_subdirectory(examples)

# development code. It requires cython, and only builds with certain versions of cython. 
add_subdirectory(dev)

