################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

## Strict warning level
if (MSVC)
    # Use the highest warning level for visual studio.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")

else()
    foreach(COMPILER C CXX)
        set(CMAKE_COMPILER_WARNINGS)
        # use -Wall for gcc and clang
        list(APPEND CMAKE_COMPILER_WARNINGS 
            -Wall
            -Wextra
            -Wcomment
            -Wendif-labels
            -Wformat
            -Winit-self
            -Wreturn-type
            -Wsequence-point
            -Wshadow
            -Wswitch
            -Wtrigraphs
            -Wundef
            -Wuninitialized
            -Wunreachable-code
            -Wunused
            -Wno-sign-compare
        )
        if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang")
            list(APPEND CMAKE_COMPILER_WARNINGS
                
                -Weverything
                #-Werror
                
                # very difficult to live without these off
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-padded
                -Wno-weak-vtables
                -Wno-exit-time-destructors
                -Wno-documentation
                
                ## TODO : profile these warnings
                -Wno-missing-prototypes
                -Wno-sign-conversion
                -Wno-conversion
                
                ## from MIOpen, don't seem to be needed in MIOpenGEMM
                #-Wno-double-promotion
                #-Wno-extra-semi
                #-Wno-float-conversion
                #-Wno-gnu-anonymous-struct
                #-Wno-gnu-zero-variadic-macro-arguments
                #-Wno-missing-braces
                #-Wno-nested-anon-types
                #-Wno-shorten-64-to-32
                #-Wno-unused-command-line-argument
                
            )
        else()
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Wno-missing-field-initializers
                -Wno-deprecated-declarations
            )
        endif()
        add_definitions(${CMAKE_COMPILER_WARNINGS})
    endforeach()
endif ()
