################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

SET(ISAAC_INCLUDE_SEARCH_PATHS
  /usr/include/isaac
  /usr/include
  /usr/local/include
  /usr/local/include/isaac
  /opt/isaac/include
  /usr/local/include
)

SET(ISAAC_LIB_SEARCH_PATHS
        /lib/
        /lib64/
        /usr/lib
        /usr/lib/isaac
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
 )


FIND_PATH(ISAAC_INCLUDE_DIR NAMES clBLAS.h PATHS ${ISAAC_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(ISAAC_LIB NAMES isaac PATHS ${ISAAC_LIB_SEARCH_PATHS})

SET(ISAAC_FOUND ON)

#    Check include files
IF(NOT ISAAC_INCLUDE_DIR)
    SET(ISAAC_FOUND OFF)
    MESSAGE(STATUS "Could not find ISAAC include. Turning ISAAC_FOUND off")
ENDIF()

#    Check libraries
IF(NOT ISAAC_LIB)
    SET(ISAAC_FOUND OFF)
    MESSAGE(STATUS "Could not find ISAAC lib. Turning ISAAC_FOUND off")
ENDIF()

IF (ISAAC_FOUND)
  IF (NOT ISAAC_FIND_QUIETLY)
    MESSAGE(STATUS "Found ISAAC libraries: ${ISAAC_LIB}")
    MESSAGE(STATUS "Found ISAAC include: ${ISAAC_INCLUDE_DIR}")
  ENDIF (NOT ISAAC_FIND_QUIETLY)
ELSE (ISAAC_FOUND)
  IF (ISAAC_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find ISAAC")
  ENDIF (ISAAC_FIND_REQUIRED)
ENDIF (ISAAC_FOUND)

MARK_AS_ADVANCED(
    ISAAC_INCLUDE_DIR
    ISAAC_LIB
    ISAAC
)
