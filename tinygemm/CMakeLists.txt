
include(GNUInstallDirs)
set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )
set( DATA_INSTALL_DIR ${CMAKE_INSTALL_FULL_DATAROOTDIR}/tinygemm )
set( CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/tinygemm )

configure_file(config.hpp.dev dev_include/tinygemm/config.hpp)
configure_file(config.hpp.install install_include/config.hpp)

file(GLOB_RECURSE source_files src/*.cpp)

add_library(tinygemm ${source_files})

target_link_libraries(tinygemm ${OPENCL_LIBRARIES})
target_link_libraries(tinygemm ${Boost_LIBRARIES} Boost::disable_autolinking)

target_include_directories (tinygemm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>
    ${OPENCL_INCLUDE_DIRS})


############################################################
# The following is cmake code to generate a config file package for Tensile
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )



configure_package_config_file(
  tinygemm-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/tinygemm-config.cmake
  INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/tinygemm-config-version.cmake
  VERSION ${tinygemm_VERSION}
  COMPATIBILITY SameMajorVersion
)

############################################################
# Library install logic

install(TARGETS tinygemm 
    EXPORT tinygemm-targets
    DESTINATION ${LIB_INSTALL_DIR})

install(DIRECTORY include/tinygemm DESTINATION ${INCLUDE_INSTALL_DIR})

install( EXPORT tinygemm-targets
  DESTINATION
    ${CONFIG_PACKAGE_INSTALL_DIR}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/tinygemm-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/tinygemm-config-version.cmake
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

