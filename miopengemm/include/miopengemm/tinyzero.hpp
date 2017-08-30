/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_TINYZEROJINX_HPP
#define GUARD_MIOPENGEMM_TINYZEROJINX_HPP

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>
#include <miopengemm/architests.hpp>
#include <miopengemm/bundle.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/timer.hpp>

namespace MIOpenGEMM
{

// For tracking the amount of work done (time, restarts, #kernels) during search for solution
class FindTracker
{

  private:
  Timer  timer;
  size_t descents{0};
  size_t kernels{0};

  public:
  void        start();
  void        incr_descents();
  void        incr_kernels();
  double      get_elapsed() const;
  size_t      get_descents() const;
  std::string get_string() const;
};

// For bundling the 4 GPU memories (a, b, c, w), and managing the copy of c if it is needed
class GpuMms
{
  public:
  std::array<cl_mem, Mem::E::N> cl_mems;

  private:
  oclutil::SafeClMem c_copy{"initialised when c_is_const"};

  public:
  GpuMms(cl_mem           a_gpu_,
         cl_mem           b_gpu_,
         cl_mem           c_gpu_,
         bool             c_is_const,
         cl_mem           workspace_gpu_,
         size_t           c_nbytes,
         cl_command_queue cq);
  cl_mem& operator[](Mem::E x);
};

// Lowest level (most basic) of MIOpenGEMM kernel search and benchmark functionality
class TinyZero
{

  // TODO : (in miogemm.hpp) miogemm class with interface to public jinx.
  public:
  TinyZero(cl_command_queue command_queue_,
           const Geometry   gg_,
           const Offsets    toff_,
           cl_mem           a_gpu_,
           cl_mem           b_gpu_,
           cl_mem           c_gpu_,
           bool             c_is_const,
           cl_mem           workspace_gpu_,
           owrite::Writer&  mowri_);

  std::vector<double> benchgemm(const HyPas& hp, const Halt& hl);
  Solution find0(const Constraints& constraint, const FindParams& find_params);

  private:
  cl_command_queue       command_queue;
  const Geometry         gg;
  const Offsets          toff;
  GpuMms                 gpum;
  const oclutil::DevInfo devinfo;
  owrite::Writer&        mowri;

  // for each of the possible kernels (copy a, copy b, etc)
  std::array<Kernel, KType::E::N> tk_kernels;

  std::array<cl_event, KType::E::N> tk_events;

  // pointers to the kernels required for a given HyPas (fewer than or as many as KType::E::N)
  std::vector<Kernel*> tk_kernels_active;

  // dependency graph of active kernels
  std::vector<std::vector<size_t>> v_wait_indices;

  double get_gflops(double timems);
  std::string get_run_times_heading();
  std::string get_run_time_string(cl_int status, double extime);
  void address_check_valid();
  void address_check_valid_and_reliable();
  void set_kern_args(const KernBlob& kblob);
  void setup_tinykernels(const kerngen::Bundle& bundle);

  Solution single_descent_find(double allotted_time,
                               const Constraints&,
                               const Halt&  core_hl,
                               FindTracker& ftrack,
                               SummStat::E  sumstat,
                               bool         warmstart,
                               size_t       warmstart_rank);

  oclutil::Result
  true_core(std::function<void(std::string)> acton, std::vector<double>& times, const Halt& hl);
};
}

#endif
