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

class Halt
{
  public:
  size_t max_runs;
  double max_time;
  Halt(size_t max_runs_, double max_time_);
  Halt() = default;
  bool halt(size_t ri, double et) const;
  std::string get_status(size_t ri, double et) const;
  std::string get_string() const;
};

class FindParams
{
  public:
  // for the outer find loop (number of descents, total time)
  Halt hl_outer;

  // for the (inner) core gemm loop (number of GEMMs, max time per kernel)
  Halt hl_core;

  SummStat::E sumstat;

  FindParams(size_t      max_descents,
             double      max_time_outer,
             size_t      max_per_kernel,
             double      max_time_core,
             SummStat::E sumstat);

  FindParams() = default;
  std::string get_string() const;
};

FindParams get_quick_find_params();
}

#endif
