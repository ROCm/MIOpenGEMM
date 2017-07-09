/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_FINDPARAMS_HPP
#define GUARD_MIOPENGEMM_FINDPARAMS_HPP

#include <string>
#include <vector>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM
{


std::string get_sumstatkey(SummStat::E sumstat);

class FindParams
{
  public:
  float       allotted_time;
  size_t    allotted_descents;
  size_t    max_n_runs_per_kernel;
  double max_time_per_kernel;
  SummStat::E sumstat;
  FindParams(float       allotted_time,
             size_t    allotted_descents,
             size_t    max_n_runs_per_kernel,
             double max_time_per_kernel,
             SummStat::E sumstat);
  FindParams() = default;
  std::string get_string() const;
};
}

#endif
