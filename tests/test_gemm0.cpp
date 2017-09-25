/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <iomanip>
#include <thread>

#include <miopengemm/apitest.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/geometries.hpp>
#include <miopengemm/hint.hpp>
#include <miopengemm/oclutil.hpp>

int main()
{

  using namespace MIOpenGEMM;

  auto                           toff = get_padding_offsets();
  owrite::Writer                 mowri(Ver::E::TERMINAL, "");
  CLHint                         devhint(0, 0);  // TODO : command line option.
  cl_command_queue_properties    cqps = 0;       // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  oclutil::CommandQueueInContext cqic(mowri, cqps, devhint, "test_geomm0");

  //std::vector<Geometry>          geometries;
  std::vector<float>             alphas;
  std::vector<float>             betas;
  std::vector<size_t>            n_runs;
  std::vector<bool>              run_accus;
  std::vector<apitest::GemmImpl> impls;
  std::vector<bool>              run_event_timers;

  const std::vector<Geometry> & geometries = get_conv_geometries();

  for (auto && x : geometries){
    (void)x;
    alphas.emplace_back(2.1);
    betas.emplace_back(0.1);
    n_runs.emplace_back(4);
    run_accus.emplace_back(true);
    impls.emplace_back(apitest::GemmImpl::XGEMM);
    run_event_timers.emplace_back(false);    
  }

  auto n_problems = geometries.size();

  std::vector<apitest::RunStats> all_runstats;
  for (auto i = 0; i < n_problems; ++i)
  {

    //std::thread([&]{
    
    std::cout << '(' << i << ')'  << '\n' << geometries[i].get_string();
    auto x = apitest::supa_gemm0<float>(cqic.command_queue,
                                        geometries[i],
                                        toff,
                                        alphas[i],
                                        betas[i],
                                        n_runs[i],
                                        run_accus[i],
                                        impls[i],
                                        run_event_timers[i],
                                        mowri);

    all_runstats.push_back(x);
    
    //}).join();
  }

  mowri << std::setw(30) << "All tests passed. Summary: " << Endl;

  mowri << std::setfill('-') << std::setw(102) << "-" << Endl;
  mowri << std::setfill(' ');

  mowri << "    m       n      k      a_t     b_t "
        << "  prec   time (usec)  tflops   numRepeats  (of " << n_problems << ") "
        << " Impl " << '\n';

  for (auto i = 0; i < n_problems; ++i)
  {
    auto& gg = geometries[i];
    mowri << std::setw(7) << gg.m << std::setw(7) << gg.n << std::setw(7) << gg.k << std::setw(7)
          << (gg.tX[Mat::E::A] ? 't' : 'n') << std::setw(7) << (gg.tX[Mat::E::B] ? 't' : 'n')
          << std::setw(8) << gg.floattype << Flush;

    auto mean_time = 1e6 * all_runstats[i].host_time / all_runstats[i].n_runs;
    auto tflops    = gg.get_gflops(1e-3 * mean_time);
    auto implstr   = apitest::get_impl_name(impls[i]);

    mowri << std::setw(12) << std::setprecision(4) << mean_time << std::setw(12) << tflops
          << std::setw(10) << all_runstats[i].n_runs << std::setw(9) << i << std::setw(8) << implstr
          << Endl;
  }

  return 0;
}
