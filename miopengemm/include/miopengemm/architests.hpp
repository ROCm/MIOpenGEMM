/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ARCHITESTS_HPP
#define GUARD_MIOPENGEMM_ARCHITESTS_HPP

#include <CL/cl.h>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/graph.hpp>

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
