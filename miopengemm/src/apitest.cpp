/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>
#include <thread>
#include <vector>

#include <miopengemm/platform.hpp>
#ifdef MIOPENGEMM_BENCH_ISAAC
#include <clBLAS.h>
#endif

#ifdef MIOPENGEMM_BENCH_CLBLAST
#include <clblast_c.h>
#endif

#include <miopengemm/accuracytests.hpp>
#include <miopengemm/apitest.hpp>
#include <miopengemm/cpugemm.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/geometries.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/programcacher.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/timer.hpp>

namespace MIOpenGEMM
{

namespace apitest
{

std::map<GemmImpl, std::string> get_impl_names()
{
  std::map<GemmImpl, std::string> x;
  x[GemmImpl::XGEMM] = "xgemm";
  x[GemmImpl::GEMM0] = "gemm0";
  x[GemmImpl::ISAAC] = "ISAAC";
  x[GemmImpl::CLB]   = "CLBlast";

  return x;
}

const std::string& get_impl_name(GemmImpl impl)
{
  const static std::map<GemmImpl, std::string> names = get_impl_names();
  if (names.find(impl) == names.end())
  {
    throw miog_error("GemmImpl key not found in map to names");
  }
  return names.at(impl);
}

RunStats::RunStats(size_t n_runs_, double host_time_, const std::vector<double>& event_times_)
  : n_runs(n_runs_), host_time(host_time_), event_times(event_times_)
{
}

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
                    const setabcw::CpuMemBundle<T>* ptr_cmb)
{

  if (get_floattype_char<T>() != gg.floattype)
  {
    throw miog_error(
      "Incompatible float types in supa_geomm0 (between alpha, beta and gg.floattype). ");
  }

  std::vector<double> event_timer_times;
  double              sum_event_times = 0;

  mowri << "\n******   Implementation: ";
  switch (impl)
  {
  case GemmImpl::XGEMM: mowri << "MIOpenGEMM's xgemm"; break;
  case GemmImpl::GEMM0: mowri << "MIOpenGEMM's gemm0"; break;
  case GemmImpl::CLB: mowri << "CLBlast"; break;
  case GemmImpl::ISAAC: mowri << "Isaac"; break;
  }
  mowri << ".   ******" << Endl;

  std::unique_ptr<setabcw::CpuMemBundle<T>> local_cmb;
  if (ptr_cmb == nullptr)
  {
    local_cmb.reset(new setabcw::CpuMemBundle<T>({gg}, toff));
    ptr_cmb = &(*local_cmb);
  }

  auto memsize = [&gg, &toff](Mat::E emat) {
    size_t ms = get_mat_memsize(gg, toff, emat);
    return ms;
  };

  std::vector<T> c_mem0(memsize(Mat::E::C));

  std::memcpy(c_mem0.data(), ptr_cmb->r_mem[Mat::E::C], memsize(Mat::E::C));

  if (beta >= 0 && beta <= 0)  // beta == 0, insert some nans to test this special case
  {
    int counter = 0;
    for (auto& x : c_mem0)
    {
      if (++counter % 2 == 0)
      {
        x = std::numeric_limits<T>::quiet_NaN();
      }
    }
  }

#ifdef MIOPENGEMM_BENCH_CLBLAST
  auto clblast_layout = gg.isColMajor ? CLBlastLayoutColMajor : CLBlastLayoutRowMajor;
  auto clblast_atrans = gg.tX[Mat::E::A] ? CLBlastTransposeYes : CLBlastTransposeNo;
  auto clblast_btrans = gg.tX[Mat::E::B] ? CLBlastTransposeYes : CLBlastTransposeNo;
#else
  if (impl == GemmImpl::CLB)
  {
    throw miog_error("MIOpenGEMM not build with CLBlast, see ccmake options to build with it");
  }
#endif

#ifdef MIOPENGEMM_BENCH_ISAAC
  auto clblas_layout = gg.isColMajor ? clblasColumnMajor : clblasRowMajor;
  auto clblas_atrans = gg.tX[Mat::E::A] ? clblasTrans : clblasNoTrans;
  auto clblas_btrans = gg.tX[Mat::E::B] ? clblasTrans : clblasNoTrans;
#else
  if (impl == GemmImpl::ISAAC)
  {
    throw miog_error("MIOpenGEMM not build with ISAAC, see ccmake options to build with it");
  }
#endif

  // .............................. set up GPU memories ..................................

  std::array<cl_mem, Mat::E::N> dev_mem;
  cl_mem dev_w = nullptr;

  for (auto x : {Mat::E::A, Mat::E::B, Mat::E::C})
  {

    if (sizeof(T) * ptr_cmb->v_mem[x]->size() < memsize(x))
    {
      std::stringstream errm;
      errm << "Size as determined by CPU is smaller than memsize, for x = " << Mat::M().name[x]
           << ", with size from CPU = " << sizeof(T) * ptr_cmb->v_mem[x]->size()
           << " and memsize = " << memsize(x);
      throw miog_error(errm.str());
    }
  }

  for (auto x : {Mat::E::A, Mat::E::B})
  {

    oclutil::cl_set_buffer_from_command_queue(
      dev_mem[x], queue, CL_MEM_READ_WRITE, memsize(x), nullptr, "(runem)", true);

    oclutil::cl_enqueue_write_buffer(queue,
                                     dev_mem[x],
                                     CL_TRUE,
                                     0,
                                     memsize(x),
                                     ptr_cmb->r_mem[x],
                                     0,
                                     nullptr,
                                     nullptr,
                                     "(runem)",
                                     true);
  }

  oclutil::cl_set_buffer_from_command_queue(
    dev_mem[Mat::E::C], queue, CL_MEM_READ_ONLY, memsize(Mat::E::C), nullptr, "(runem)", true);

  oclutil::cl_enqueue_write_buffer(queue,
                                   dev_mem[Mat::E::C],
                                   CL_TRUE,
                                   0,
                                   memsize(Mat::E::C),
                                   c_mem0.data(),
                                   0,
                                   nullptr,
                                   nullptr,
                                   "(runem)",
                                   true);

  auto w_mem_size = get_total_workspace(gg, toff) * gg.derived.float_size_bytes;
  if (w_mem_size > 0)
  {
    oclutil::cl_set_buffer_from_command_queue(
      dev_w, queue, CL_MEM_READ_WRITE, w_mem_size, nullptr, " ", true);
  }
  // ............................... GPU memories setup  ..................................

  size_t n_warmup  = 1;
  size_t n_to_time = n_runs - n_warmup;
  Timer  timer;
  int    xgemm_ID = -1;

  for (size_t run_i = 0; run_i < n_runs; ++run_i)
  {
    if (run_i == n_warmup)
    {
      timer.start();
    }

    // TODO : use shared_ptr from a smart cl_event.
    cl_event  base_gemmevent = nullptr;
    auto      use_cl_event   = (run_i < n_warmup || run_event_timer || run_i == n_to_time);
    cl_event* ptr_gemmevent  = (use_cl_event) ? &base_gemmevent : nullptr;

    // MIOpenGEMM GEMM
    if (impl == GemmImpl::XGEMM)
    {

      auto result = xgemm<T>(gg.isColMajor,
                             gg.tX[Mat::E::A],
                             gg.tX[Mat::E::B],
                             gg.m,
                             gg.n,
                             gg.k,
                             alpha,
                             dev_mem[Mat::E::A],
                             toff.offsets[Mem::E::A],
                             gg.ldX[Mat::E::A],
                             dev_mem[Mat::E::B],
                             toff.offsets[Mem::E::B],
                             gg.ldX[Mat::E::B],
                             beta,
                             dev_mem[Mat::E::C],
                             toff.offsets[Mem::E::C],
                             gg.ldX[Mat::E::C],
                             dev_w,
                             toff.offsets[Mem::E::W],
                             gg.wSpaceSize,
                             &queue,
                             0,
                             nullptr,
                             ptr_gemmevent,
                             xgemm_ID);

      xgemm_ID = result.ID;
    }

    else if (impl == GemmImpl::GEMM0)
    {

      gemm0<T>(gg.isColMajor,
               gg.tX[Mat::E::A],
               gg.tX[Mat::E::B],
               gg.m,
               gg.n,
               gg.k,
               alpha,
               dev_mem[Mat::E::A],
               toff.offsets[Mem::E::A],
               gg.ldX[Mat::E::A],
               dev_mem[Mat::E::B],
               toff.offsets[Mem::E::B],
               gg.ldX[Mat::E::B],
               beta,
               dev_mem[Mat::E::C],
               toff.offsets[Mem::E::C],
               gg.ldX[Mat::E::C],
               &queue,
               0,
               nullptr,
               ptr_gemmevent);
    }

#ifdef MIOPENGEMM_BENCH_CLBLAST
    else if (impl == GemmImpl::CLB)
    {
      CLBlastStatusCode status = CLBlastSgemm(clblast_layout,
                                              clblast_atrans,
                                              clblast_btrans,
                                              gg.m,
                                              gg.n,
                                              gg.k,
                                              alpha,
                                              dev_mem[Mat::E::A],
                                              toff.offsets[Mem::E::A],
                                              gg.ldX[Mat::E::A],
                                              dev_mem[Mat::E::B],
                                              toff.offsets[Mem::E::B],
                                              gg.ldX[Mat::E::B],
                                              beta,
                                              dev_mem[Mat::E::C],
                                              toff.offsets[Mem::E::C],
                                              gg.ldX[Mat::E::C],
                                              &queue,
                                              ptr_gemmevent);
      (void)status;
    }
#endif

#ifdef MIOPENGEMM_BENCH_ISAAC
    else if (impl == GemmImpl::ISAAC)
    {

      clblasStatus isaac_status = clblasSgemm(clblas_layout,
                                              clblas_atrans,
                                              clblas_btrans,
                                              gg.m,
                                              gg.n,
                                              gg.k,
                                              alpha,
                                              dev_mem[Mat::E::A],
                                              toff.offsets[Mem::E::A],
                                              gg.ldX[Mat::E::A],
                                              dev_mem[Mat::E::B],
                                              toff.offsets[Mem::E::B],
                                              gg.ldX[Mat::E::B],
                                              beta,
                                              dev_mem[Mat::E::C],
                                              toff.offsets[Mem::E::C],
                                              gg.ldX[Mat::E::C],
                                              1,
                                              &queue,
                                              0,
                                              nullptr,
                                              ptr_gemmevent);

      (void)isaac_status;
    }
#endif

    else
    {
      std::stringstream ss;
      ss << "unrecognised GemmImpl " << static_cast<int>(impl) << '\n';
      throw miog_error(ss.str());
    }

    if (use_cl_event)
    {
      clWaitForEvents(1, ptr_gemmevent);
    }

    // perform accuracy test if first run and accuracy test requested
    if (run_i == 0 && run_accu)
    {

      std::stringstream infoss;
      infoss << apitest::get_impl_name(impl) << '\n' << gg.get_string() << '\n';
      if (impl == GemmImpl::GEMM0 || impl == GemmImpl::XGEMM)
      {
        auto id = get_cacher().get_ID_from_geom(gg, get_beta_type(beta), &queue);
        infoss << get_cacher().hyper_params[id].get_string();
      }

      // read from device
      std::vector<T> c_from_device(memsize(Mat::E::C));
      cl_event       readevent;
      oclutil::cl_enqueue_read_buffer(queue,
                                      dev_mem[Mat::E::C],
                                      CL_TRUE,
                                      0,
                                      memsize(Mat::E::C),
                                      c_from_device.data(),
                                      0,
                                      nullptr,
                                      &readevent,
                                      "read from device",
                                      true);

      // perform GEMM on CPU
      std::vector<T> c_cpu(c_mem0);
      cpugemm::gemm<T>(gg,
                       toff,
                       ptr_cmb->r_mem[Mat::E::A],
                       ptr_cmb->r_mem[Mat::E::B],
                       c_cpu.data(),
                       alpha,
                       beta,
                       mowri);

      // perform absolute GEMM on CPU

      std::vector<T> A_abs(memsize(Mat::E::A) / sizeof(T));
      std::memcpy(A_abs.data(), ptr_cmb->r_mem[Mat::E::A], memsize(Mat::E::A));
      for (auto& x : A_abs)
      {
        x = std::abs(x);
      }

      std::vector<T> B_abs(memsize(Mat::E::B) / sizeof(T));
      std::memcpy(B_abs.data(), ptr_cmb->r_mem[Mat::E::B], memsize(Mat::E::B));
      for (auto& x : B_abs)
      {
        x = std::abs(x);
      }

      std::vector<T> C_abs(memsize(Mat::E::C) / sizeof(T));
      std::memcpy(C_abs.data(), c_mem0.data(), memsize(Mat::E::C));
      for (auto& x : C_abs)
      {
        x = std::abs(x);
      }

      cpugemm::gemm<T>(
        gg, toff, A_abs.data(), B_abs.data(), C_abs.data(), std::abs(alpha), std::abs(beta), mowri);

      // make sure the readevent is complete and then release it.
      oclutil::cl_wait_for_events(1, &readevent, "waiting for read from device", true);
      oclutil::cl_release_event(readevent, "device read", true);

      // compare cpu and gpu results
      accuracytests::elementwise_compare(gg,
                                         toff,
                                         c_mem0.data(),         // before,
                                         c_cpu.data(),          // after cpu
                                         c_from_device.data(),  // after gpu
                                         C_abs.data(),          // after abs on cpu
                                         infoss.str(),
                                         mowri);

      mowri << '\n';
    }

    if (run_event_timer && run_i >= n_warmup)
    {

      size_t t_start;
      oclutil::cl_set_event_profiling_info(*ptr_gemmevent,
                                           CL_PROFILING_COMMAND_START,
                                           sizeof(size_t),
                                           &t_start,
                                           nullptr,
                                           "apitest : " + get_impl_name(impl),
                                           true);

      size_t t_end;
      oclutil::cl_set_event_profiling_info(
        *ptr_gemmevent, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end, nullptr, "(runem)", true);

      event_timer_times.push_back(1e-6 * (t_end - t_start));
      sum_event_times += event_timer_times.back();

      if (run_i == n_warmup)
      {
        mowri << "cl event \"gflops\" : [";
      }

      if (run_i < 10 || n_to_time - run_i < 10)
      {
        mowri << ' ' << gg.get_gflops(1e-3 * event_timer_times.back()) << ' ' << Flush;
      }
      else if (run_i == 10)
      {
        mowri << "   ...   " << Flush;
      }

      if (run_i == n_to_time)
      {
        mowri << ']';
        mowri << "\nmean event time : " << sum_event_times / n_to_time << Endl;
      }
    }

    if (use_cl_event)
    {
      oclutil::cl_release_event(*ptr_gemmevent, "from VeriGEMM", true);
    }
  }

  // overall timer
  auto   t_total   = 1000 * timer.get_elapsed();
  double mean_time = t_total / n_to_time;
  double gflops    = gg.get_gflops(1e-3 * mean_time);
  mowri << "total host time : " << t_total << " [ms] total runs : " << n_to_time << Endl;
  mowri << "mean  host time : " << mean_time << "   mean gflops : " << gflops << Endl;

  if (run_event_timer)
  {
    mowri << "mean time diff (event - host) " << mean_time - sum_event_times / n_to_time << " [ms] "
          << Endl;
  }

  mowri << '\n';
  // Clean-up
  for (auto x : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    oclutil::cl_release_mem_object(dev_mem[x], "release memory in apitest", true);
  }
  oclutil::cl_release_mem_object(dev_w, "release w in apitest", true);

  return RunStats(n_to_time, t_total / 1000., event_timer_times);
}

template RunStats supa_gemm0(cl_command_queue&                   queue,
                             const Geometry&                     gg,
                             const Offsets&                      toff,
                             const float                         alpha,
                             const float                         beta,
                             size_t                              n_runs,
                             bool                                run_accu,
                             GemmImpl                            impl,
                             bool                                run_event_timer,
                             owrite::Writer&                     mowri,
                             const setabcw::CpuMemBundle<float>* ptr_cmb);

template RunStats supa_gemm0(cl_command_queue&                    queue,
                             const Geometry&                      gg,
                             const Offsets&                       toff,
                             const double                         alpha,
                             const double                         beta,
                             size_t                               n_runs,
                             bool                                 run_accu,
                             GemmImpl                             impl,
                             bool                                 run_event_timer,
                             owrite::Writer&                      mowri,
                             const setabcw::CpuMemBundle<double>* ptr_cmb);

std::string get_summary_deepstyle(const std::vector<Geometry>& geometries,
                                  const std::vector<RunStats>& all_runstats,
                                  const std::vector<GemmImpl>& impls,
                                  const std::vector<float>&    betas)
{

  std::stringstream ss;

  auto n_problems = geometries.size();

  ss << std::setfill('-') << std::setw(102) << "-" << '\n'
     << std::setfill(' ') << "    m       n      k      a_t     b_t "
     << "  prec   time (usec)  tflops "
     << "  numRepeats  (of " << n_problems << ") "
     << " Impl   beta" << '\n';

  for (auto i = 0; i < n_problems; ++i)
  {
    auto& gg = geometries[i];
    ss << std::setw(7) << gg.m << std::setw(7) << gg.n << std::setw(7) << gg.k << std::setw(7)
       << (gg.tX[Mat::E::A] ? 't' : 'n') << std::setw(7) << (gg.tX[Mat::E::B] ? 't' : 'n')
       << std::setw(8) << gg.floattype;

    auto mean_time = 1e6 * all_runstats[i].host_time / all_runstats[i].n_runs;
    auto tflops    = gg.get_gflops(1e-3 * mean_time);
    auto implstr   = get_impl_name(impls[i]);

    ss << std::setw(12) << std::setprecision(4) << mean_time << std::setw(12) << tflops
       << std::setw(10) << all_runstats[i].n_runs << std::setw(9) << i << std::setw(8) << implstr
       << std::setw(7) << betas[i] << '\n';
  }

  return ss.str();
}
}
}
