cmake_minimum_required(VERSION 3.0)

# Default installation path
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/x86_64-w64-mingw32" CACHE PATH "Installation path")
else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Installation path")
endif()

project(MIOpenGEMM)

set(${PROJECT_NAME}_VERSION 0.1)
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 1)

#Where to find FindOpenCL.cmake (etc)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
##Use FindOpenCL.cmake to find library and header file
find_package( OpenCL REQUIRED )

find_package(PythonInterp REQUIRED)

# What I would like here : user sets the above directory, and at configuration time the file system is generated.
# Currently, I assume that the tree structure of the writing dir is already setup. 
# add_definitions(-DDIR_FOR_WRITING="/home/james/MIOpenGEMMout")

option(BUILD_SHARED_LIBS "Build shared library" ON)

include(EnableCompilerWarnings)

add_subdirectory(MIOpenGEMM)
add_subdirectory(tests)

# development code. It requires cython (and only builds with certain versions. You probably want to comment this out.
#add_subdirectory(dev)
