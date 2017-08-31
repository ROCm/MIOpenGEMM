SET(CLBLAST_INCLUDE_SEARCH_PATHS
  #$ENV{CLBLAST_HOME}
  #$ENV{CLBLAST_HOME}/include
  #/usr/include/openblas
  #/usr/include
  #/usr/include/openblas-base
  #/usr/local/include
  #/usr/local/include/openblas
  #/usr/local/include/openblas-base
  #/opt/CLBLAST/include
  /home/james/nugteren/clblast/install/include
)

SET(CLBLAST_LIB_SEARCH_PATHS
        #/opt/CLBLAST/lib
        #$ENV{CLBLAST}cd
        #$ENV{CLBLAST}/lib
        #$ENV{CLBLAST_HOME}
        #$ENV{CLBLAST_HOME}/lib
        #/lib/
        #/lib/openblas-base
        #/lib64/
        #/usr/lib
        #/usr/lib/openblas-base
        #/usr/lib64
        #/usr/local/lib
        #/usr/local/lib64
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
