/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_FINDPARAMS_HPP
#define GUARD_MIOPENGEMM_FINDPARAMS_HPP

#include <string>
#include <vector>

namespace MIOpenGEMM
{

enum SummaryStat
{
  Mean = 0,
  Median,
  Max,
  nSumStatKeys
};

std::string get_sumstatkey(SummaryStat sumstat);

class FindParams
{
  public:
  float       allotted_time;
  unsigned    allotted_descents;
  unsigned    n_runs_per_kernel;
  SummaryStat sumstat;
  FindParams(float       allotted_time,
             unsigned    allotted_descents,
             unsigned    n_runs_per_kernel,
             SummaryStat sumstat);
  FindParams() = default;
  std::string get_string() const;
};
}

#endif
