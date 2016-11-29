SET(CYTHONGEN_CPP ${CMAKE_CURRENT_BINARY_DIR}/cythongen/pytinygemm.cpp)
SET(CYTHON_PYX ${CMAKE_CURRENT_SOURCE_DIR}/pytinygemm.pyx)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cythongen)



include_directories("../../tinygemm/include")


add_library(pytinygemm SHARED  ${CYTHONGEN_CPP})
#SHARED

SET_TARGET_PROPERTIES(pytinygemm PROPERTIES LINKER_LANGUAGE CXX PREFIX "")

#How and when to cythonise
add_custom_command(
  OUTPUT ${CYTHONGEN_CPP}
  DEPENDS ${CYTHON_PYX} #${JEXE1_NAME}
  COMMAND cython --cplus ${CYTHON_PYX} -o ${CYTHONGEN_CPP}
)

target_include_directories(pytinygemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(pytinygemm LINK_PUBLIC tinygemm  ${OpenCL_LIBRARY})


execute_process(COMMAND python-config --cflags OUTPUT_VARIABLE CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE "-Wall" "" CFLAGS ${CFLAGS})
string(REPLACE "-Wextra" "" CFLAGS ${CFLAGS})
string(REPLACE "-pedantic" "" CFLAGS ${CFLAGS})

SET(CMAKE_CXX_FLAGS  " ${CFLAGS} --std=c++11")

