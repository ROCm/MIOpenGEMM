/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#ifndef GUARD_MIOPENGEMM_PROGRAMSES_HPP
#define GUARD_MIOPENGEMM_PROGRAMSES_HPP

#include <CL/cl.h>
#include <algorithm>
#include <vector>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{

class Program
{

  public:
  cl_device_id device_id;
  cl_context   context;
  KernBlob     kblob;
  cl_program   clprog;

  //                 execution times
  size_t             t_start;
  size_t             t_end;
  std::vector<float> v_times;

  public:
  Program(cl_device_id, cl_context);
  Program() : Program(nullptr, nullptr) {}
  //bool            update_needed(const KernBlob&);
  oclutil::Result update(const KernBlob&, owrite::Writer&, const std::string& build_options);
  ~Program();
  Program& operator=(const Program&) = delete;
  Program& operator=(Program&&) = default;
  void     update_times(const cl_event&);
  void     reset_times();

  private:
  void try_release();
};

class BasePrograms
{

  public:
  double extime;
  
  using AllKernArgs = std::vector<std::vector<std::pair<size_t, const void*>>>;

  std::array<Program, KType::E::N> programs;
  std::vector<size_t>                act_inds;
  std::vector<std::vector<size_t>> v_wait_indices;

  owrite::Writer * ptr_mowri;
  
  void reset_times();
  // This function will
  // (1) create a vector of cl_kernels from programs indexed by act_inds.
  // (2) create a vector of cl_events for each kernel, the last one might be user provided.
  // (3) for each index :
  //     (3.1) make std::vector of cl_events which block this kernel (see kernel.hpp)
  //     (3.2) set the arguments of the kernel.
  //     (3.3) enqueue the kernel (see kernel.hpp)
  // (4) if update_times, update program times (use act_inds).

  virtual oclutil::Result run(const cl_command_queue&,
                              const AllKernArgs&,
                              cl_uint         n_user_wait_list,
                              const cl_event* user_wait_list,
                              bool            update_times, 
                              cl_event *      ptr_user_event) = 0;

  // This function will update
  // (1) act_inds
  // (2) programs and
  // (3) v_wait_indices
  void update(const std::vector<KernBlob>&);

  //void update_times();
  
  size_t get_n_active() { return act_inds.size(); }
  BasePrograms(const cl_device_id&, const cl_context&, owrite::Writer & mowri_);
  
  BasePrograms() = default;
};

class VerbosePrograms : public BasePrograms
{
  public:
  virtual oclutil::Result run(const cl_command_queue&,
                              const AllKernArgs&,
                              cl_uint         n_user_wait_list,
                              const cl_event* user_wait_list,
                              bool            update_times, 
                              cl_event *      ptr_user_event);
                              
  VerbosePrograms(const cl_device_id& id, const cl_context& ctxt, owrite::Writer & mowri_):
  BasePrograms(id, ctxt, mowri_){}
  
  VerbosePrograms() = default;
  
  
};

}
#endif

