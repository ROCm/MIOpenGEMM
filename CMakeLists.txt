cmake_minimum_required(VERSION 3.0)
project(TINYGEMM)

#Where to find FindOpenCL.cmake (etc)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

##Use FindOpenCL.cmake to find library and header file
find_package( OpenCL REQUIRED )

find_package(PythonInterp REQUIRED)

# What I would like here : user sets the above directory, and at configuration time the file system is generated.
# Currently, I assume that the tree structure of the writing dir is already setup. 
# add_definitions(-DDIR_FOR_WRITING="/home/james/tinygemmout")

option(BUILD_SHARED_LIBS ON)

add_subdirectory(tinygemm)
add_subdirectory(testsexamples)

# development code. It requires cython (and only builds with certain versions. You probably want to comment this out.
add_subdirectory(dev)
