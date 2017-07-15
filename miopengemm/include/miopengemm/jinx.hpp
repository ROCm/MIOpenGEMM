#ifndef GUARD_MIOPENGEMM_JINX_HPP
#define GUARD_MIOPENGEMM_JINX_HPP

/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
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


  
namespace MIOpenGEMM
{

// Simple stop-watch
class Timer{
  
  private:
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;  
  
  public:
  void start();
  double get_elapsed() const;
};

class FindTracker{
  
  private:
  Timer timer;  
  size_t descents{0};
  size_t kernels{0};
  
  public:
  void start();
  double get_elapsed() const;
  void incr_descents();
  void incr_kernels();
  
  size_t get_descents() const;
  
  std::string get_string() const;
  
};


class GpuMms
{
  private:
  std::array<cl_mem, Mem::E::N> cl_mems;
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


class Jinx
{

  // TODO : miogemm class with interface to public jinx. 
  public:
  Jinx(cl_command_queue command_queue_,
       const Geometry   gg_,
       const Offsets    toff_,
       cl_mem           a_gpu_,
       cl_mem           b_gpu_,
       cl_mem           c_gpu_,
       bool             c_is_const,
       cl_mem           workspace_gpu_,
       owrite::Writer&  mowri_);

  void benchgemm(const HyPas& hp, const Halt & hl);
  Solution find(const Constraints& constraint, const FindParams& find_params);

  ///////////////////////////////////////////////////////////////////////////////////////////

  private:
  cl_command_queue       command_queue;
  const Geometry         gg;
  const Offsets          toff;
  GpuMms                 gpum;
  const oclutil::DevInfo devinfo;
  owrite::Writer&        mowri;

  
  // (for single_descent_find) while generating, compiling and
  // benchmarking kernels, we will keep track of the
  // fastest found thus far
  std::vector<Kernel>                                         tk_kernels;
  std::vector<Kernel*>                                        tk_kernels_active;
  std::vector<std::vector<size_t>>                            v_wait_indices;
  

  double get_gflops(double timems);

  std::string get_run_times_heading();
  std::string get_run_time_string(cl_int status);
  
  void address_check_valid();
  void address_check_valid_and_reliable();
  void set_kern_args(const KernUses& type);

  std::string get_run_time_string(cl_int status, double extime);
      
  bool refresh_needed(KType::E type, const HyPas& new_hp, const DerivedParams& new_dp);
  oclutil::Result refresh_kernel(const KernBlobg& ks, const HyPas& hp, const DerivedParams& dp);
  oclutil::Result setup_tinykernels(const HyPas& hp, const kerngen::Bundle& bundle);
  Solution single_descent_find(double allotted_time, const Constraints&, const Halt& core_hl, FindTracker & ftrack, const FindParams &); //TODO FindParams should not be needed

  oclutil::Result true_core(std::function<void(double, std::string)> acton, const Halt & hl);

};
}

#endif
