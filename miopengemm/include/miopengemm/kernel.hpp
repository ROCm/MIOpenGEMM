/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNEL_HPP
#define GUARD_MIOPENGEMM_KERNEL_HPP

#include <CL/cl.h>
#include <algorithm>
#include <vector>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{

// TODO RAII cl_*

class Kernel
{

  public:
  cl_device_id device_id;
  cl_context   context;
  KernBlob     kblob;
  cl_program   clprog;

  cl_event*    ptr_event;
  cl_kernel    clkern;

  std::string  hash;

  // stores (the most recent of max_n_runs or fewer) execution time
  size_t             t_start;
  size_t             t_end;
  std::vector<float> v_times;


  private:
  void try_release();


  public:
  Kernel(
    cl_device_id       device_id_,
    cl_context         context_,
    cl_event*          ptr_event_,
    const std::string& hash_);

  Kernel() : Kernel(nullptr, nullptr, nullptr, "default constructed Kernel") {}

  bool            update_needed(const KernBlob&);
  oclutil::Result update_program(const KernBlob&, owrite::Writer&, const std::string& build_options);
  void            update_kernel();

  ~Kernel();

  Kernel& operator=(const Kernel&) = delete;
  Kernel& operator=(Kernel&&) = default;

  bool is_set();
  void set_kernel_args(const std::vector<std::pair<size_t, const void*>>& arg_sizes_values);

  void update_times();
  void reset_times();
};

oclutil::Result run_kernels(cl_command_queue                 command_queue,
                            std::vector<Kernel*>             ptr_kernels,
                            std::vector<std::vector<size_t>> v_wait_indices,
                            cl_uint                          n_user_wait_list,
                            const cl_event*                  user_wait_list);
// bool use_event,
// cl_event * ptr_event);
}

#endif
