/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <sstream>
#include <miopengemm/enums.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
namespace MIOpenGEMM
{

Halt::Halt(std::array<size_t, Xtr::E::N> runs, std::array<double, Xtr::E::N> time)
{

  max_runs = runs[Xtr::E::MAX];
  min_runs = runs[Xtr::E::MIN];

  max_time = time[Xtr::E::MAX];
  min_time = time[Xtr::E::MIN];

  if (max_time <= 0)
  {
    throw miog_error("max_time should be strictly positive, in Halt constructor");
  }

  if (max_time < min_time)
  {
    throw miog_error("max_time < min_time, in Halt constructor (not allowed)");
  }

  if (max_runs == 0)
  {
    throw miog_error("max_runs should be strictly positive, in Halt constructor");
  }

  if (max_runs < min_runs)
  {
    throw miog_error("max_runs < min_runs, in Halt constructor (not allowed)");
  }
}

std::string Halt::get_status(size_t ri, double et) const
{
  if (halt(ri, et))
  {
    return "(HALT)";
  }
  std::stringstream ss;
  ss << "{" << min_time << " <<< @t=" << et << "[s] <<< " << max_time << "}   "
     << "{" << min_runs << " <<< @i=" << ri << " <<< " << max_runs << "}   ";
  return ss.str();
}

bool Halt::halt(size_t ri, double et) const
{
  // if under one of the mins or under all the maxes, continue
  if (ri < min_runs || et < min_time || (ri < max_runs && et < max_time))
  {
    return false;
  }
  return true;
}

std::string Halt::get_string() const
{
  std::stringstream ss;
  ss << '(' << min_time << " time " << max_time << ") (" << min_runs << " runs " << max_runs << ')';
  return ss.str();
}

std::vector<std::string> get_sumstatkeys_basic()
{
  std::vector<std::string> ssv(SummStat::E::N, "unset");
  ssv[SummStat::E::MEAN]   = "MEAN";
  ssv[SummStat::E::MEDIAN] = "MEDIAN";
  ssv[SummStat::E::MAX]    = "MAX";
  for (size_t i = 0; i < SummStat::E::N; ++i)
  {
    if (ssv[i] == "unset")
    {
      throw miog_error("one of the keys has not been set for sumstatkey");
    }
  }

  return ssv;
}

const std::vector<std::string>& get_sumstatkeys()
{
  static const std::vector<std::string> sumstatkeys = get_sumstatkeys_basic();
  return sumstatkeys;
}

std::string get_sumstatkey(SummStat::E sumstat)
{
  if (sumstat >= SummStat::E::N)
  {
    throw miog_error("unrecognised sumstat key in get_sumstatkey");
  }
  return get_sumstatkeys()[sumstat];
}

FindParams::FindParams(std::array<size_t, Xtr::E::N> descents,
                       std::array<double, Xtr::E::N> time_outer,
                       std::array<size_t, Xtr::E::N> per_kernel,
                       std::array<double, Xtr::E::N> time_core,
                       SummStat::E sumstat_)

  : hl_outer(descents, time_outer), hl_core(per_kernel, time_core), sumstat(sumstat_)
{
}

std::string FindParams::get_string() const
{
  std::stringstream ss;
  ss << "(OUTER)   " << hl_outer.get_string() << "(INNER)   " << hl_core.get_string()
     << "(SUMSTAT) " << get_sumstatkey(sumstat);
  return ss.str();
}

FindParams get_at_least_n_seconds(double seconds)
{
  std::array<size_t, Xtr::E::N> descents{{0, 100000}};
  std::array<double, Xtr::E::N> time_outer{{seconds, std::min(seconds * 2, seconds + 0.1)}};
  std::array<size_t, Xtr::E::N> per_kernel{{0, 4}};
  std::array<double, Xtr::E::N> time_core{{0, .1}};  // no need to run more than .13 seconds.
  SummStat::E sumstat = SummStat::E::MAX;
  return FindParams(descents, time_outer, per_kernel, time_core, sumstat);
}

FindParams get_at_least_n_restarts(size_t restarts)
{
  std::array<size_t, Xtr::E::N> descents{{restarts, restarts}};
  std::array<double, Xtr::E::N> time_outer{{0, 10000000.}};
  std::array<size_t, Xtr::E::N> per_kernel{{0, 5}};
  std::array<double, Xtr::E::N> time_core{{0, .1}};  // no need to run more than .1 seconds.
  SummStat::E sumstat = SummStat::E::MEDIAN;
  return FindParams(descents, time_outer, per_kernel, time_core, sumstat);
}
}
