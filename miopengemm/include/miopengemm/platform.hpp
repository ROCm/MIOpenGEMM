/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_PLATFORM_HPP
#define GUARD_MIOPENGEMM_PLATFORM_HPP

#ifdef __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif

#endif
