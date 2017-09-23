/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#ifndef GUARD_MIOPENGEMM_PROGRAMSES_HPP
#define GUARD_MIOPENGEMM_PROGRAMSES_HPP

#include <algorithm>
#include <memory>
#include <vector>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/platform.hpp>

namespace MIOpenGEMM
{

using AllKernArgs = std::vector<std::vector<std::pair<size_t, const void*>>>;

class SafeCLProgram
{
  public:
  cl_program clprog = nullptr;

  ~SafeCLProgram()
  {
    if (clprog)
    {
      oclutil::cl_release_program(clprog, "~Program", true);
    }
  }
};

class KernelTime
{
  public:
  size_t              t_start;
  size_t              t_end;
  std::vector<double> v_times;
  void                update_times(const cl_event&);
  void                reset_times();
};

class KernelTimes
{
  public:
  std::array<KernelTime, KType::E::N> ktimes;
  double extime;
  void   reset_times();
};

class Program
{

  public:
  cl_device_id device_id;
  cl_context   context;
  KernBlob     kblob;

  std::shared_ptr<SafeCLProgram> sclp;
  Program(cl_device_id, cl_context);
  Program() : Program(nullptr, nullptr) {}
  oclutil::Result update(const KernBlob&, owrite::Writer&, const std::string& build_options);
};

class Programs
{

  public:
  std::array<Program, KType::E::N> programs;
  std::vector<size_t>              act_inds;
  std::vector<std::vector<size_t>> v_wait_indices;
  owrite::Writer*                  ptr_mowri;

  // This function will
  // (1) create a vector of cl_kernels from programs indexed by act_inds.
  // (2) create a vector of cl_events for each kernel except the last one.
  // (3) for each kernel k (index in act_inds):
  //     (3.1) make std::vector of cl_events which block k
  //     (3.2) set the arguments of the k
  //     (3.3) enqueue k
  // (4) if update_times, update program times (use act_inds).
  oclutil::Result run(const cl_command_queue&,
                      const AllKernArgs&,
                      cl_uint         n_user_wait_list,
                      const cl_event* user_wait_list,
                      KernelTimes*    ptr_ktimes,
                      cl_event*       ptr_user_event,
                      bool            debug_mode) const;

  // This function will update
  // (1) act_inds
  // (2) programs and
  // (3) v_wait_indices
  oclutil::Result update(const std::vector<KernBlob>&);

  size_t get_n_active() const { return act_inds.size(); }
  Programs(const cl_device_id&, const cl_context&, owrite::Writer& mowri_);

  Programs() = default;

  // Programs & operator=(const Programs &) = delete;
  // Programs (const Programs &)  = delete;
  // Programs (Programs &&)  = default;
  // Programs& operator=(Programs &&)  = default;
};
}
#endif
