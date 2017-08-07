/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_SOLUTION_HPP
#define GUARD_MIOPENGEMM_SOLUTION_HPP

#include <chrono>
#include <map>
#include <string>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM
{

// Note : A Solution is only valid for a fixed Geometry
class Solution
{
  public:
  // Geometry on which this solution was obtained
  Geometry geometry;

  // GPU time required to run Solution
  double extime;

  // OpenCL kernel strings
  std::vector<KernBlob> v_tgks;

  // hyper-parameters of kernel(s) in v_tgks
  HyPas hypas;

  // Info about device used to find solution
  oclutil::DevInfo devinfo;

  // Constraints imposed while searching for this solution
  Constraints constraints;

  Solution(const Geometry&,
           double extime,
           const std::vector<KernBlob>&,
           const HyPas&,
           const oclutil::DevInfo&,
           const Constraints&);

  // return a string summarising the Geometry, (a request from MIOpen)
  std::string get_networkconfig_string() const;

  std::string get_cache_entry_string() const;
};
}

#endif
