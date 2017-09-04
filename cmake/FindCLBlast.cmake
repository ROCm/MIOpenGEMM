################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

SET(CLBLAST_INCLUDE_SEARCH_PATHS
  /usr/include/isaac
  /usr/include
  /usr/local/include
  /usr/local/include/clblast
  /opt/isaac/include
  /usr/local/include  
  /home/james/nugteren/clblast/install/include
)

SET(CLBLAST_LIB_SEARCH_PATHS
        /lib/
        /lib64/
        /usr/lib
        /usr/lib/clblast
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /home/james/nugteren/clblast/install/lib
)

FIND_PATH(CLBLAST_INCLUDE_DIR NAMES clblast.h PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(CLBLAST_LIB NAMES clblast PATHS ${CLBLAST_LIB_SEARCH_PATHS})

SET(CLBLAST_FOUND ON)

#    Check include files
IF(NOT CLBLAST_INCLUDE_DIR)
    SET(CLBLAST_FOUND OFF)
    MESSAGE(STATUS "Could not find CLBLAST include. Turning CLBLAST_FOUND off")
ENDIF()

#    Check libraries
IF(NOT CLBLAST_LIB)
    SET(CLBLAST_FOUND OFF)
    MESSAGE(STATUS "Could not find CLBLAST lib. Turning CLBLAST_FOUND off")
ENDIF()

IF (CLBLAST_FOUND)
  IF (NOT CLBLAST_FIND_QUIETLY)
    MESSAGE(STATUS "Found CLBLAST libraries: ${CLBLAST_LIB}")
    MESSAGE(STATUS "Found CLBLAST include: ${CLBLAST_INCLUDE_DIR}")
  ENDIF (NOT CLBLAST_FIND_QUIETLY)
ELSE (CLBLAST_FOUND)
  IF (CLBLAST_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find CLBLAST")
  ENDIF (CLBLAST_FIND_REQUIRED)
ENDIF (CLBLAST_FOUND)

MARK_AS_ADVANCED(
    CLBLAST_INCLUDE_DIR
    CLBLAST_LIB
    CLBLAST
)
