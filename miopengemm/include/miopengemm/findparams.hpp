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
  size_t min_runs;
  double max_time;
  double min_time;

  Halt(std::array<size_t, Xtr::E::N> runs, std::array<double, Xtr::E::N> time);

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

  FindParams(std::array<size_t, Xtr::E::N> descents,
             std::array<double, Xtr::E::N> time_outer,
             std::array<size_t, Xtr::E::N> per_kernel,
             std::array<double, Xtr::E::N> time_core,
             SummStat::E sumstat);

  FindParams() = default;
  std::string get_string() const;
};

FindParams get_at_least_n_seconds(double seconds);
FindParams get_at_least_n_restarts(size_t restarts);
}

#endif
