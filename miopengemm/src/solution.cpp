/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <sstream>
#include <string>
#include <string>
#include <vector>

#include <miopengemm/error.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/kernelcache.hpp>

namespace MIOpenGEMM
{

std::string Solution::get_networkconfig_string() const
{
  return geometry.get_networkconfig_string();
}


std::string Solution::get_cache_entry_string() const
{
  return MIOpenGEMM::get_cache_entry_string({devinfo.device_name, constraints, geometry}, hypas);
  
}

}

