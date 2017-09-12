/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ARCHITESTS_HPP
#define GUARD_MIOPENGEMM_ARCHITESTS_HPP

#ifdef __APPLE__
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

class Stat
{
  public:
  bool        is_good;
  std::string msg;
  Stat(const oclutil::DevInfo&, const DerivedParams&, const Geometry&, const HyPas&);
};
}
}

#endif
