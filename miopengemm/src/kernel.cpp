/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <iomanip>
#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/oclutil.hpp>

namespace MIOpenGEMM
{

Kernel::Kernel(cl_device_id       device_id_,
               cl_context         context_,
               cl_event*          ptr_event_,
               const std::string& hash_)

  :
    device_id(device_id_),
    context(context_),
    ptr_event(ptr_event_),
    clprog(nullptr),
    clkern(nullptr),
    hash(hash_)
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


void Kernel::update_kernel(){

  oclutil::cl_create_kernel(clkern,
                          clprog,
                          kblob.fname.c_str(),
                          "setting kernel in oclutil::Result",
                          true);
  
}

// TODO : add build_options flag.
oclutil::Result
Kernel::update_program(const KernBlob& ks, owrite::Writer& mowri, const std::string& build_options)
{

  oclutil::Result oclr;

  try_release();
  kblob = ks;
  mowri << "compiling " << KType::M.name[kblob.e_ktype] << ". " << Flush;

  auto start = std::chrono::high_resolution_clock::now();
  oclr = oclutil::cl_set_program(
    context,
    device_id,
    kblob.kernstr,
    clprog,
    build_options,
    mowri,
    false);
  
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

void Kernel::set_kernel_args(const std::vector<std::pair<size_t, const void*>>& arg_sizes_values)
{
  oclutil::cl_set_kernel_args(clkern, arg_sizes_values, "Kernel::set_kernel_args", true);
}


void Kernel::update_times()
{

  oclutil::cl_set_event_profiling_info(
    *ptr_event, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start, nullptr, "u_times", true);

  oclutil::cl_set_event_profiling_info(
    *ptr_event, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end, nullptr, "in update_times", true);

  v_times.push_back(1e-6 * (t_end - t_start));
}

void Kernel::reset_times()
{
  t_start = 0;
  t_end   = 0;
  v_times.resize(0);
}

oclutil::Result run_kernels(cl_command_queue                 command_queue,
                            std::vector<Kernel*>             ptr_kernels,
                            std::vector<std::vector<size_t>> v_wait_indices,
                            cl_uint                          n_user_wait_list,
                            const cl_event*                  user_wait_list)
{

  for (size_t k_ind = 0; k_ind < ptr_kernels.size(); ++k_ind)
  {

    std::vector<cl_event> clevent_waits;

    for (cl_uint i = 0; i < n_user_wait_list; ++i)
    {
      clevent_waits.emplace_back(user_wait_list[i]);
    }

    for (auto& evi : v_wait_indices[k_ind])
    {
      // see `cl-events' comment at bottom
      clevent_waits.emplace_back(*(ptr_kernels[evi]->ptr_event));
    }

    const cl_event* event_wait_list = clevent_waits.size() == 0 ? nullptr : clevent_waits.data();

    auto oclr = oclutil::cl_enqueue_ndrange_kernel(command_queue,
                                            ptr_kernels[k_ind]->clkern,
                                            1,
                                            NULL,
                                            &ptr_kernels[k_ind]->kblob.global_work_size,
                                            &ptr_kernels[k_ind]->kblob.local_work_size,
                                            clevent_waits.size(), //num_events_in_wait_list,
                                            event_wait_list,
                                            ptr_kernels[k_ind]->ptr_event,
                                            "Kernel::enqueue",
                                            false);
                                            
    // see `in-series' comment at bottom

    if (oclr.success == CL_SUCCESS)
    {
      // good
    }

    else
    {
      oclr.message += "(run_kernels, kernel " + std::to_string(k_ind) + ")";
      return oclr;
    }
  }
  return {};
}
}
