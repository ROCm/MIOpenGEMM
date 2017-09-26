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

  std::vector<Geometry> geometries = get_conv_geometries();
  geometries.resize(20);
  geometries.emplace_back("tC0_tA1_tB1_colMaj0_m400_n500_k600_lda1002_ldb1004_ldc1008_ws0_f32");

  // std::vector<Geometry> geometries = get_deepbench(0);

  auto n_problems = geometries.size();

  for (auto i = 0; i < n_problems; ++i)
  {
    alphas.emplace_back(2.0);

    if (i % 3 == 0)
    {
      betas.emplace_back(0);
    }
    else if (i % 3 == 1)
    {
      betas.emplace_back(1);
    }
    else
    {
      betas.emplace_back(0.5);
    }

    n_runs.emplace_back(3 + 3 * (i % 3 == 2));

    run_accus.emplace_back(true);

    if (i % 3 == 0)
    {
      impls.emplace_back(apitest::GemmImpl::XGEMM);
    }

    else
    {
      impls.emplace_back(apitest::GemmImpl::GEMM0);
    }

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
  size_t block_size = 3;
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
