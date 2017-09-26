/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_APITEST00_HPP
#define GUARD_MIOPENGEMM_APITEST00_HPP

#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/platform.hpp>
#include <miopengemm/setabcw.hpp>

namespace MIOpenGEMM
{

namespace apitest
{

enum class GemmImpl
{
  XGEMM = 0,  // MIOpenGEMM
  GEMM0,      // MIOpenGEMM
  ISAAC,      // Isaac
  CLB,        // CLBlast
};

const std::string& get_impl_name(GemmImpl);

class RunStats
{
  public:
  size_t              n_runs;
  double              host_time;
  std::vector<double> event_times;
  RunStats(size_t n_runs_, double host_time_, const std::vector<double>& event_times_);
  RunStats() = default;
};

template <typename T>
RunStats supa_gemm0(cl_command_queue&               queue,
                    const Geometry&                 gg,
                    const Offsets&                  toff,
                    const T                         alpha,
                    const T                         beta,
                    size_t                          n_runs,
                    bool                            run_accu,
                    GemmImpl                        impl,
                    bool                            run_event_timer,
                    owrite::Writer&                 mowri,
                    const setabcw::CpuMemBundle<T>* ptr_cmb);  // if nullptr, done in function.

std::string get_summary_deepstyle(const std::vector<Geometry>& geometries,
                                  const std::vector<RunStats>& all_runstats,
                                  const std::vector<GemmImpl>& impls,
                                  const std::vector<float>&    betas);
}
}

#endif
