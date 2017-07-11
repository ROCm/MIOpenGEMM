/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/enums.hpp>
namespace MIOpenGEMM
{

std::vector<std::string> get_sumstatkey()
{
  std::vector<std::string> ssv(SummStat::E::N, "unset");
  ssv[SummStat::E::MEAN]   = "Mean";
  ssv[SummStat::E::MEDIAN] = "Median";
  ssv[SummStat::E::MAX]    = "Max";
  for (size_t i = 0; i < SummStat::E::N; ++i)
  {
    if (ssv[i] == "unset")
    {
      throw miog_error("one of the keys has not been set for sumstatkey");
    }
  }

  return ssv;
}

const std::vector<std::string> sumstatkey = get_sumstatkey();

std::string get_sumstatkey(SummStat::E sumstat)
{

  if (sumstat >= SummStat::E::N)
  {
    throw miog_error("unrecognised sumstat key in get_sumstatkey");
  }
  return sumstatkey[sumstat];
}

/* TODO floats to doubles */
FindParams::FindParams(float       allotted_time_,
                       size_t    allotted_descents_,
                       size_t    max_n_runs_per_kernel_,
                       double    max_time_per_kernel_,
                       SummStat::E sumstat_)
  : allotted_time(allotted_time_),
    allotted_descents(allotted_descents_),
    max_n_runs_per_kernel(max_n_runs_per_kernel_),
    max_time_per_kernel(max_time_per_kernel_),
    sumstat(sumstat_)
{

  if (allotted_time <= 0)
  {
    throw miog_error("allotted_time should be strictly positive, in FindParams constructor");
  }

  if (allotted_descents == 0)
  {
    throw miog_error("allotted_descents should be strictly positive, in "
                     "FindParams constructor");
  }
}

std::string FindParams::get_string() const
{
  std::stringstream ss;
  ss << "allotted time: " << allotted_time << " allotted_descents: " << allotted_descents
     << " max_n_runs_per_kernel: " << max_n_runs_per_kernel << " max_time_per_kernel: " << max_time_per_kernel << "  sumstat: " << get_sumstatkey(sumstat)
     << std::endl;
  return ss.str();
}



FindParams get_quick_find_params(){

  float                   allotted_time       = 0.003;
  size_t                allotted_iterations = 1;
  size_t                max_n_runs_per_kernel   = 1;
  double max_time_per_kernel = 1e12;
  SummStat::E sumstat = SummStat::E::MEDIAN;

  return FindParams(
    allotted_time, allotted_iterations, max_n_runs_per_kernel, max_time_per_kernel, sumstat);
}


}
