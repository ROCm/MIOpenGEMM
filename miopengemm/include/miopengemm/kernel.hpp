/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 *******************************************************************************/

#ifndef GUARD_MIOPENGEMM_KERNEL_HPP
#define GUARD_MIOPENGEMM_KERNEL_HPP

#include <CL/cl.h>
#include <algorithm>
#include <vector>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/openclutil.hpp>
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

  // stores (the most recent of n_runs) execution time
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

  openclutil::OpenCLResult update(const KernelString& ks, outputwriting::OutputWriter& mowri);

  ~Kernel();

  Kernel& operator=(const Kernel&) = default;

  bool is_set();
  void set_kernel_args(std::vector<std::pair<size_t, const void*>> arg_sizes_values);

  openclutil::OpenCLResult enqueue(cl_uint         num_events_in_wait_list,
                                   const cl_event* event_wait_list);
  openclutil::OpenCLResult enqueue();

  void update_times();

  void reset_times();
};
}

#endif
