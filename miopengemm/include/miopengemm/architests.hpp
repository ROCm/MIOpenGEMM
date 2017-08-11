/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ARCHITESTS_HPP
#define GUARD_MIOPENGEMM_ARCHITESTS_HPP

#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/hyperparams.hpp>

namespace MIOpenGEMM
{
namespace architests
{

std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue,
                                                          const derivedparams::DerivedParams&,
                                                          const Geometry& gg,
                                                          const hyperparams::HyperParams&);
}
}

#endif
