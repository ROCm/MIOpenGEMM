/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <sstream>
#include <string>
#include <string>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{

Solution::Solution(const Geometry&              gg,
                   double                       extime_,
                   const std::vector<KernBlob>& v1,
                   const HyPas&                 hp,
                   const oclutil::DevInfo&      di,
                   const Constraints&           co)
  : geometry(gg), extime(extime_), v_tgks(v1), hypas(hp), devinfo(di), constraints(co)
{
}

std::string Solution::get_networkconfig_string() const
{
  return geometry.get_networkconfig_string();
}

std::string Solution::get_cache_entry_string() const
{
  return MIOpenGEMM::get_cache_entry_string(
    {devinfo.identifier, constraints, redirection::get_canonical(geometry)},
    hypas,
    redirection::get_is_not_canonical(geometry));
}
}
