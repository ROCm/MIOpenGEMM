/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <iomanip>
#include <miopengemm/error.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/oclutil.hpp>

namespace MIOpenGEMM
{

Kernel::Kernel(cl_command_queue command_queue_, const std::string& hash_)
  : command_queue(command_queue_), clprog(nullptr), clkern(nullptr), hash(hash_)
{
}

void Kernel::try_release()
{

  if (clprog != nullptr)
  {
    oclutil::cl_release_program(clprog, "Kernel Destructor", true);
  }
  if (clkern != nullptr)
  {
    oclutil::cl_release_kernel(clkern, "Kernel Destructor", true);
  }
}

oclutil::Result Kernel::update(const KernBlob& ks, owrite::Writer& mowri)
{

  oclutil::Result oclr;

  // for (unsigned i = 0; i < 2000; ++i)
  //{

  try_release();
  kblob = ks;
  mowri << "compiling " << KType::M.name[kblob.e_ktype] << ". " << Flush;

  auto start = std::chrono::high_resolution_clock::now();

  oclr = oclutil::cl_set_program_and_kernel(
    command_queue, kblob.kernstr, kblob.fname, clprog, clkern, mowri, false);

  auto                         end             = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms           = end - start;
  float                        elapsed_seconds = fp_ms.count();

  if (oclr.fail())
  {
    mowri << "Failed in " << std::setprecision(3) << elapsed_seconds << std::setprecision(6)
          << " [s]" << Endl;
  }
  else
  {
    mowri << "Done in " << std::setprecision(3) << elapsed_seconds << std::setprecision(6) << " [s]"
          << Endl;
  }

  //}
  return oclr;
}

Kernel::~Kernel() { try_release(); }

bool Kernel::is_set() { return (clprog != nullptr && clkern != nullptr); }

bool Kernel::update_needed(const KernBlob& kb_new)
{
  if (!is_set())
  {
    return true;
  }
  auto no_change = (kb_new.kernstr == kblob.kernstr);
  bool change    = !no_change;
  return change;
}

void Kernel::set_kernel_args(std::vector<std::pair<size_t, const void*>> arg_sizes_values)
{
  oclutil::cl_set_kernel_args(clkern, arg_sizes_values, "Kernel::set_kernel_args", true);
}

oclutil::Result Kernel::enqueue(cl_uint num_events_in_wait_list, const cl_event* event_wait_list)
{

  return oclutil::cl_enqueue_ndrange_kernel(command_queue,
                                            clkern,
                                            1,
                                            NULL,
                                            &kblob.global_work_size,
                                            &kblob.local_work_size,
                                            num_events_in_wait_list,
                                            event_wait_list,
                                            &clevent,
                                            "Kernel::enqueue",
                                            false);
}

oclutil::Result Kernel::enqueue() { return enqueue(0, nullptr); }

void Kernel::update_times()
{

  oclutil::cl_set_event_profiling_info(
    clevent, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start, nullptr, "u_times", true);

  oclutil::cl_set_event_profiling_info(
    clevent, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end, nullptr, "in update_times", true);

  v_times.push_back(1e-6 * (t_end - t_start));
}

void Kernel::reset_times()
{
  t_start = 0;
  t_end   = 0;
  v_times.resize(0);
}
}
