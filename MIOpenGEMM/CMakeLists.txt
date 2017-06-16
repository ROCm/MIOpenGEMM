SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic " )

include(InstallSymlinks)
include(GNUInstallDirs)
set( MIOpenGEMM_INSTALL_DIR MIOpenGEMM)
set( BIN_INSTALL_DIR ${MIOpenGEMM_INSTALL_DIR}/${CMAKE_INSTALL_BINDIR} )
set( LIB_INSTALL_DIR ${MIOpenGEMM_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${MIOpenGEMM_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR} )
set( DATA_INSTALL_DIR ${MIOpenGEMM_INSTALL_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/MIOpenGEMM )
set( CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/MIOpenGEMM )

include(CreatePackage)
create_package(
    NAME MIOpenGEMM
    DESCRIPTION "A tool for generating OpenCL matrix multiplication (GEMM) kernels"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG ${LIB_INSTALL_DIR}
)

option(BUILD_DEV OFF)

if(BUILD_DEV)
configure_file(config.hpp.dev dev_include/MIOpenGEMM/config.hpp)
else()
configure_file(config.hpp.install dev_include/MIOpenGEMM/config.hpp)
endif()

file(GLOB_RECURSE source_files src/*.cpp)

add_library(MIOpenGEMM ${source_files})

target_link_libraries(MIOpenGEMM ${OPENCL_LIBRARIES})

target_include_directories (MIOpenGEMM PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>
    ${OPENCL_INCLUDE_DIRS})

target_include_directories (MIOpenGEMM SYSTEM INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

############################################################
# The following is cmake code to generate a config file package for Tensile
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( PackageConfigHelpers )



configure_package_config_file(
  MIOpenGEMM-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MIOpenGEMM-config.cmake
  INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR CONFIG_PACKAGE_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MIOpenGEMM-config-version.cmake
  VERSION ${MIOpenGEMM_VERSION}
  COMPATIBILITY SameMajorVersion
)

############################################################
# Library install logic

install(TARGETS MIOpenGEMM 
    EXPORT MIOpenGEMM-targets
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR})

install(DIRECTORY include/MIOpenGEMM DESTINATION ${INCLUDE_INSTALL_DIR})

install( EXPORT MIOpenGEMM-targets
  DESTINATION
    ${CONFIG_PACKAGE_INSTALL_DIR}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/MIOpenGEMM-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/MIOpenGEMM-config-version.cmake
  DESTINATION
    ${CONFIG_PACKAGE_INSTALL_DIR} )

install(DIRECTORY
  python
  opencl
  DESTINATION ${DATA_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.cl"
)

install_symlink_subdir(${MIOpenGEMM_INSTALL_DIR})

