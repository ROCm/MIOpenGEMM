/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <iomanip>
#include <mutex>
#include <thread>

#include <miopengemm/apitest.hpp>
#include <miopengemm/geometries.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hint.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/timer.hpp>
#include <miopengemm/randomutil.hpp>

// NOTE : compiling kernels is the bottleneck here (when OpenBLAS is used for CPU),
// precomputing CPU results will not accelerate significantly.

int main()
{

  using namespace MIOpenGEMM;

  auto                           toff = get_padding_offsets();
  owrite::Writer                 mowri(Ver::E::TERMINAL, "");
  CLHint                         devhint(0, 0);
  cl_command_queue_properties    cqps = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  oclutil::CommandQueueInContext cqic(mowri, cqps, devhint, "test_gemm0");

  std::vector<float>             alphas;
  std::vector<float>             betas;
  std::vector<size_t>            n_runs;
  std::vector<bool>              run_accus;
  std::vector<apitest::GemmImpl> impls;
  std::vector<bool>              run_event_timers;

  std::vector<Geometry> geometries;
  size_t n_problems = 1000;
  
  RandomUtil rutil(1011);


  for (auto i = 0; i < n_problems; ++i)
  {
    
    
    auto r1 = rutil.get_from_range(3);
    switch (r1){
      case 0 : alphas.emplace_back(2.0);
      case 1 : alphas.emplace_back(-2.0);
      case 2 : alphas.emplace_back(0.5);
    }


    auto r2 = rutil.get_from_range(3);
    switch (r2){
      case 0 : betas.emplace_back(0.0);
      case 1 : betas.emplace_back(1.0);
      case 2 : betas.emplace_back(0.5);
    }

    auto r3 = rutil.get_from_range(2);
    switch (r3){
      case 0 : n_runs.emplace_back(2);
      case 1 : n_runs.emplace_back(3);
    }

    run_accus.emplace_back(true);

    auto r4 = rutil.get_from_range(2);
    switch (r4){
      case 0 : impls.emplace_back(apitest::GemmImpl::XGEMM);
      case 1 : impls.emplace_back(apitest::GemmImpl::GEMM0);
    }

    int m = 1 + rutil.get_from_range(1000);
    int n = 1 + rutil.get_from_range(1000);
    int k = 1 + rutil.get_from_range(1000);
    geometries.emplace_back(get_padded_geometry<float>(true, false, false, false, m, n, k, 0));
    
    run_event_timers.emplace_back(false);
  }

  std::vector<apitest::RunStats> all_runstats;
  all_runstats.resize(n_problems);

  std::vector<std::thread> threads;

  const setabcw::CpuMemBundle<float> cmb(geometries, toff);

  auto run = [&](size_t i) {
    mowri << '(' << i << " of " << n_problems << ')' << Endl;
    mowri << '\n' << Flush << geometries[i].get_string();

    auto x = apitest::supa_gemm0<float>(cqic.command_queue,
                                        geometries[i],
                                        toff,
                                        alphas[i],
                                        betas[i],
                                        n_runs[i],
                                        run_accus[i],
                                        impls[i],
                                        run_event_timers[i],
                                        mowri,
                                        &cmb);

    all_runstats[i] = x;
  };

  Timer timer;
  timer.start();

  // join the threads is blocks of block_size.
  size_t block_size = 1;
  for (auto i = 0; i < n_problems / block_size; ++i)
  {
    for (auto j = i * block_size; j < (i + 1) * block_size; ++j)
    {
      threads.emplace_back(std::thread(run, j));
    }
    for (auto j = i * block_size; j < (i + 1) * block_size; ++j)
    {
      threads[j].join();
    }
  }
  for (auto j = block_size * (n_problems / block_size); j < n_problems; ++j)
  {
    threads.emplace_back(std::thread(run, j));
  }
  for (auto j = block_size * (n_problems / block_size); j < n_problems; ++j)
  {
    threads[j].join();
  }

  mowri << "\ntime elapsed : " << timer.get_elapsed();

  mowri << std::setw(30) << "All tests passed. Summary: " << Endl;

  mowri << apitest::get_summary_deepstyle(geometries, all_runstats, impls, betas);

  return 0;
}
