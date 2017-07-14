/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNEL_HPP
#define GUARD_MIOPENGEMM_KERNEL_HPP

#include <CL/cl.h>
#include <algorithm>
#include <vector>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{

class Kernel
{

  public:
  cl_command_queue command_queue;
  KernelString     tgk_strings;

  // used for getting performance of kernel
  cl_event clevent;

  // stores (the most recent of max_n_runs or fewer) execution time
  size_t             t_start;
  size_t             t_end;
  std::vector<float> v_times;

  private:
  cl_program clprog;

  public:
  cl_kernel clkern;

  public:
  std::string hash;

  private:
  void try_release();
  void set_kernel_arg(cl_uint arg_index, size_t arg_size, const void* arg_value);

  public:
  Kernel(cl_command_queue command_queue_, const std::string& hash_);

  Kernel() : Kernel(nullptr, "default constructed Kernel") {}

  oclutil::Result update(const KernelString& ks, owrite::Writer& mowri);

  ~Kernel();

  Kernel& operator=(const Kernel&) = default;

  bool is_set();
  void set_kernel_args(std::vector<std::pair<size_t, const void*>> arg_sizes_values);

  oclutil::Result enqueue(cl_uint num_events_in_wait_list, const cl_event* event_wait_list);
  oclutil::Result enqueue();

  void update_times();

  void reset_times();
};
}

#endif
