/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <sstream>
#include <miopengemm/enums.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
namespace MIOpenGEMM
{

Halt::Halt(size_t max_runs_, double max_time_) : max_runs(max_runs_), max_time(max_time_)
{
  if (max_time <= 0)
  {
    throw miog_error("max_time should be strictly positive, in Halt constructor");
  }

  if (max_runs == 0)
  {
    throw miog_error("max_runs should be strictly positive, in Halt constructor");
  }
}

std::string Halt::get_status(size_t ri, double et) const
{
  if (halt(ri, et))
  {
    return "(HALT)";
  }
  std::stringstream ss;
  ss << "@t=" << et << " [s] (<" << max_time << " [s]) @i=" << ri << " (<" << max_runs << " )";
  return ss.str();
}

bool Halt::halt(size_t ri, double et) const { return (ri >= max_runs || et >= max_time); }

std::string Halt::get_string() const
{
  std::stringstream ss;
  ss << "max_time=" << max_time << " max_runs=" << max_runs;
  return ss.str();
}

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

FindParams::FindParams(

  size_t      max_descents,
  double      max_time_outer,
  size_t      max_per_kernel,
  double      max_time_core,
  SummStat::E sumstat_)

  : hl_outer(max_descents, max_time_outer),
    hl_core(max_per_kernel, max_time_core),
    sumstat(sumstat_)
{
}

std::string FindParams::get_string() const
{
  std::stringstream ss;
  ss << "(OUTER)   " << hl_outer.get_string() << "(INNER)   " << hl_core.get_string()
     << "(SUMSTAT) " << get_sumstatkey(sumstat);
  return ss.str();
}

FindParams get_quick_find_params()
{
  size_t      max_descents   = 1;
  double      max_time_outer = 0.003;
  size_t      max_per_kernel = 1;
  double      max_time_core  = 1e12;
  SummStat::E sumstat        = SummStat::E::MEDIAN;

  return FindParams(max_descents, max_time_outer, max_per_kernel, max_time_core, sumstat);
}
}
