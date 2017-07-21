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

//// Solution discovery time statistics
// class DiscoStamp
//{

// public:
//// time in find when found, run # of single_descent when found. (o)uter
// double o_time;
// size_t o_it;
//// time in single_descent when found, kernel # within single_descent when found. (i)nner
// double i_time;
// size_t i_it;
//// timestamp (date) when found
// std::string date;
//// time recordered to run the kernel
// double extime;

// DiscoStamp(double o_t, size_t o_i, double i_t, size_t i_i, std::string d, double et);

//// TODO
// std::string get_string() const;
//};

// A Solution is only valid for a fixed Geometry
class Solution
{

  public:
  // Geometry on which this solution was obtained
  Geometry geometry;
  // discovery time of solution
  // DiscoStamp disco;
  // timestamp (date) when found
  std::string date;
  double      extime;
  // the OpenCL kernel strings
  std::vector<KernBlob> v_tgks;
  // hyper-parameters of kernel(s) in v_tgks
  HyPas            hypas;
  oclutil::DevInfo devinfo;
  Constraints      constraints;

  Solution(const Geometry&              gg,
           const std::string&           date_,
           double                       extime_,
           const std::vector<KernBlob>& v1,
           const HyPas&                 hp,
           const oclutil::DevInfo&      di,
           const Constraints&           co)
    : geometry(gg),
      date(date_),
      extime(extime_),
      v_tgks(v1),
      hypas(hp),
      devinfo(di),
      constraints(co)
  {
  }

  // return a string summarising the Geometry,
  // less offsets (a request from MLOpen)
  std::string get_networkconfig_string() const;

  //// return a string describing the hyper parameters
  // std::string get_hyper_param_string() const;

  std::string get_cache_entry_string() const;
};
}

#endif
