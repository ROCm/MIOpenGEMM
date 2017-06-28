/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 *******************************************************************************/

#ifndef GUARD_MIOPENGEMM_SOLUTION_HPP
#define GUARD_MIOPENGEMM_SOLUTION_HPP

#include <chrono>
#include <map>
#include <string>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM
{

class SolutionStatistics
{
  public:
  // the median time and flop/s recorded with
  // the(se) kernel(s) on the specified benchmarked problem
  float median_benchmark_time;
  float median_benchmark_gflops;

  // the time in seconds at which this solution
  // was discovered  (from start of descent)
  float solution_discovery_time;

  // timestamp (date) when found
  std::string date;

  FindParams find_params;

  SolutionStatistics(float             median_benchmark_time,
                     float             median_benchmark_gflops,
                     float             solution_discovery_time,
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

  // the kernels of which the solution is comprised
  std::vector<KernelString> v_tgks;

  std::string hyper_param_string;

  openclutil::OpenCLDeviceInfo devinfo;

  std::string constraints_string;

  Solution(const Geometry&                  geometry_,
           SolutionStatistics               tgss_,
           const std::vector<KernelString>& v_tgks_,
           std::string                      hyper_param_string_,
           openclutil::OpenCLDeviceInfo     devinfo_,
           std::string                      constraints_string_)
    : geometry(geometry_),
      statistics(tgss_),
      v_tgks(v_tgks_),
      hyper_param_string(hyper_param_string_),
      devinfo(devinfo_),
      constraints_string(constraints_string_)
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
