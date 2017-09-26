/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include <CL/cl.h>

#include <miopengemm/apitest.hpp>
#include <miopengemm/cpugemm.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/geometries.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/timer.hpp>

int main()
{

  using namespace MIOpenGEMM;
  // Tracking issues observed with DeepBench problems.
  std::array<std::vector<Geometry>, 10> problems;

  // Problem geometries in Isaac.
  // https://github.com/ptillet/isaac/issues/26
  // DeepBench memory issues (on ROCm 1.6)
  problems[static_cast<int>(apitest::GemmImpl::ISAAC)] = {
    {"tC0_tA0_tB0_colMaj1_m35_n8457_k4096_lda35_ldb4096_ldc35_ws0_f32"},
    {"tC0_tA0_tB0_colMaj1_m35_n8457_k2048_lda35_ldb2048_ldc35_ws0_f32"},
    {"tC0_tA0_tB1_colMaj1_m2048_n7133_k2048_lda2048_ldb7133_ldc2048_ws0_f32"},
    {"tC0_tA0_tB1_colMaj1_m3072_n7435_k1024_lda3072_ldb7435_ldc3072_ws0_f32"},
    {"tC0_tA0_tB1_colMaj1_m4096_n7133_k4096_lda4096_ldb7133_ldc4096_ws0_f32"}};

  // Problem geometries in CLBlast. Either,
  // https://github.com/CNugteren/CLBlast/issues/185
  // (1) Excessive memory (on ROCm 1.6) or
  // (2) Memory out of bounds (on ROCm 1.6).
  problems[static_cast<int>(apitest::GemmImpl::CLB)] = {
    {"tC0_tA1_tB0_colMaj1_m512_n8_k500000_lda500000_ldb500000_ldc512_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m1024_n8_k500000_lda500000_ldb500000_ldc1024_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m512_n16_k500000_lda500000_ldb500000_ldc512_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m1024_n16_k500000_lda500000_ldb500000_ldc1024_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m4096_n16_k4096_lda4096_ldb4096_ldc4096_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m2048_n32_k2048_lda2048_ldb2048_ldc2048_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m6144_n16_k2048_lda2048_ldb2048_ldc6144_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m8448_n16_k2816_lda2816_ldb2816_ldc8448_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m7680_n16_k2560_lda2560_ldb2560_ldc7680_ws0_f32"},
    {"tC0_tA0_tB0_colMaj1_m35_n8457_k2048_lda35_ldb2048_ldc35_ws0_f32"},
    {"tC0_tA1_tB0_colMaj1_m35_n8457_k2048_lda2048_ldb2048_ldc35_ws0_f32"}};

  problems[static_cast<int>(apitest::GemmImpl::XGEMM)] = {};
  problems[static_cast<int>(apitest::GemmImpl::GEMM0)] = {};

  bool run_event_timers = true;

  auto                        toff = get_padding_offsets();
  owrite::Writer              mowri(Ver::E::TERMINAL, "");
  CLHint                      devhint(0, 0);
  cl_command_queue_properties cqps = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  if (run_event_timers)
  {
    cqps = cqps | CL_QUEUE_PROFILING_ENABLE;
  }
  oclutil::CommandQueueInContext cqic(mowri, cqps, devhint, "test_gemm0");

  std::vector<float>             all_betas;
  std::vector<Geometry>          all_geometries;
  std::vector<apitest::RunStats> all_runstats;
  std::vector<apitest::GemmImpl> all_impls;

  bool                  run_deepbench = false;
  std::vector<Geometry> geometries;
  if (run_deepbench)
  {
    geometries = get_deepbench(0);
  }

  // custom geometries.
  else
  {
    // std::vector<Geometry> geometries;
    // for (auto x : {699, 700, 701, 702, 703, 704}){
    // geometries.push_back(get_squareNN_geometry<float>(x));
    //}
    // geometries = {{"tC0_tA0_tB0_colMaj0_m13_n13_k13_lda13_ldb13_ldc13_ws0_f32"}};
    // geometries = {get_squareNN_geometry<float>(28)};
    geometries = {//{512, 16, 512, false, false, 0, 'f'},
                  //{512, 17, 512, false, false, 0, 'f'},
                  //{512, 18, 512, false, false, 0, 'f'},
                  {1024, 1024, 1024, false, false, 0, 'f'},
                  {510, 510, 510, false, false, 0, 'f'}};
  }

  Timer timer;
  timer.start();

  setabcw::CpuMemBundle<float> cmb(geometries, toff);

  for (unsigned i = 0; i < geometries.size(); ++i)
  {
    const Geometry& gg = geometries[i];

    float alpha = 1.0;
    float beta  = 1.0;

    // number of runs with timer (based on DeepBench timing method).
    size_t n_to_time =
      std::min<size_t>(1500, std::max<size_t>(std::ceil(1e11 / (2 * gg.m * gg.k * gg.n)), 2));
    bool run_accu = false;

    for (auto&& impl : {apitest::GemmImpl::GEMM0,
                        apitest::GemmImpl::XGEMM,
                        apitest::GemmImpl::CLB,
                        apitest::GemmImpl::ISAAC})
    {

      auto impl_int = static_cast<int>(impl);
      if (std::find(problems[impl_int].begin(), problems[impl_int].end(), gg) ==
          problems[impl_int].end())
      {

        auto x = apitest::supa_gemm0<float>(cqic.command_queue,
                                            geometries[i],
                                            toff,
                                            alpha,
                                            beta,
                                            n_to_time + 1,
                                            run_accu,
                                            impl,
                                            run_event_timers,
                                            mowri,
                                            &cmb);
        all_geometries.push_back(gg);
        all_runstats.push_back(x);
        all_impls.push_back(impl);
        all_betas.push_back(beta);
      }
    }
  }

  mowri << "\ntime elapsed : " << timer.get_elapsed() << Endl;

  mowri << apitest::get_summary_deepstyle(all_geometries, all_runstats, all_impls, all_betas);

  return 0;
}
