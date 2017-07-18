
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

class SolutionStatistics
{
  public:
  // the median time and flops recorded with
  // the(se) kernel(s) on the specified benchmarked problem
  double seconds;
  double gflops;

  // the time in seconds at which this solution
  // was discovered  (from start of descent)
  double discovery;

  // timestamp (date) when found
  std::string date;

  FindParams find_params;

  SolutionStatistics(double            seconds,
                     double            gflops,
                     double            discovery,
                     std::string       date,
                     const FindParams& find_params);
  SolutionStatistics(std::string cache_string);

  SolutionStatistics() = default;

  std::string get_string() const;
};

// A Solution is only valid for a fixed Geometry
class Solution
{

  public:
  // Geometry on which this solution was obtained
  Geometry geometry;

  SolutionStatistics statistics;

  // the OpenCL kernel strings
  std::vector<KernBlob> v_tgks;

  HyPas hypas;

  oclutil::DevInfo devinfo;

  Constraints constraints;

  Solution(const Geometry&              geometry_,
           SolutionStatistics           tgss_,
           const std::vector<KernBlob>& v_tgks_,
           std::string                  hyper_param_string_,
           oclutil::DevInfo             devinfo_,
           const Constraints&           constraints_)
    : geometry(geometry_),
      statistics(tgss_),
      v_tgks(v_tgks_),
      hypas(hyper_param_string_),
      devinfo(devinfo_),
      constraints(constraints_)
  {
  }

  // return a string summarising the Geometry,
  // less offsets (a request from MLOpen)
  std::string get_networkconfig_string() const;

  // return a string describing the hyper parameters
  std::string get_hyper_param_string() const;

  std::string get_cache_entry_string(std::string k_comment = "") const;
};
}

#endif
