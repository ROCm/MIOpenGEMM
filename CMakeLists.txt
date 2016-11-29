cmake_minimum_required(VERSION 3.0)
project(TINYGEMM)

#Where to find FindOpenCL.cmake (etc)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

##Use FindOpenCL.cmake to find library and header file
find_package( OpenCL REQUIRED )

# What I would like here : user sets the above directory, and at configuration time the file system is generated.
# Currently, I assume that the tree structure of the writing dir is already setup. 
add_definitions(-DDIR_FOR_WRITING="/home/james/tinygemmout")

option(BUILD_SHARED_LIBS ON)
#SET(CMAKE_CXX_COMPILER clang++)

add_subdirectory(tinygemm)
add_subdirectory(tests)
add_subdirectory(examples)

# development code. It requires cython (and only builds with certain versions. You probably want to comment this out.
add_subdirectory(dev)


#//Path to a file.
#OpenCL_INCLUDE_DIR:PATH=/opt/AMDAPPSDK-3.0/include

#//Path to a library.
#OpenCL_LIBRARY:FILEPATH=/home/james/opencl/lib/x86_64/libOpenCL.so




#//OpenCL header file path
#OPENCL_INCLUDE_DIRS:PATH=/opt/AMDAPPSDK-3.0/include

#//OpenCL dynamic library path
#OPENCL_LIBRARIES:FILEPATH=/opt/AMDAPPSDK-3.0/lib/x86/libOpenCL.so
