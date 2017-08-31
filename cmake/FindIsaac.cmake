SET(ISAAC_INCLUDE_SEARCH_PATHS
  #$ENV{ISAAC_HOME}
  #$ENV{ISAAC_HOME}/include
  #/usr/include/openblas
  #/usr/include
  #/usr/include/openblas-base
  #/usr/local/include
  #/usr/local/include/openblas
  #/usr/local/include/openblas-base
  #/opt/ISAAC/include

  /usr/local/include
)

SET(ISAAC_LIB_SEARCH_PATHS
        #/opt/ISAAC/lib
        #$ENV{ISAAC}cd
        #$ENV{ISAAC}/lib
        #$ENV{ISAAC_HOME}
        #$ENV{ISAAC_HOME}/lib
        #/lib/
        #/lib/openblas-base
        #/lib64/
        #/usr/lib
        #/usr/lib/openblas-base
        #/usr/lib64
        #/usr/local/lib
        #/usr/local/lib64

        /usr/local/lib
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
