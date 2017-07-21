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

namespace MIOpenGEMM
{

std::string Solution::get_networkconfig_string() const
{
  return geometry.get_networkconfig_string();
}

std::string Solution::get_cache_entry_string() const
{
  std::stringstream cache_write_ss;
  cache_write_ss << "kc.add(\n";
  cache_write_ss << "{\"" << devinfo.identifier << "\",  // dev\n";
  cache_write_ss << "{\"" << constraints.get_string() << "\"},  // con\n";
  cache_write_ss << "{\"" << geometry.get_string() << "\"}}, // gg\n";
  cache_write_ss << "{{ // hp\n";
  cache_write_ss << "\"" << hypas.sus[Mat::E::A].get_string() << "\",\n";
  cache_write_ss << "\"" << hypas.sus[Mat::E::B].get_string() << "\",\n";
  cache_write_ss << "\"" << hypas.sus[Mat::E::C].get_string() << "\"}});\n";
  return cache_write_ss.str();
}
}
