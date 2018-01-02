/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <chrono>
#include <iomanip>
#include <sstream>
#include <miopengemm/bundle.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/programs.hpp>

namespace MIOpenGEMM
{

Program::Program(cl_device_id id, cl_context ctxt)
  : device_id(id), context(ctxt), sclp(new SafeCLProgram)
{
}

oclutil::Result
Program::update(const KernBlob& ks, owrite::Writer& mowri, const std::string& build_opts)
{

  oclutil::Result oclr;

  // no update needed
  if ((sclp->clprog != nullptr) && (ks.kernstr == kblob.kernstr))
  {
    oclr = {};
  }

  else
  {
    if (sclp->clprog != nullptr)
    {
      oclutil::cl_release_program(sclp->clprog, "update", true);
    }

    kblob = ks;
    mowri << "compiling " << KType::M().name[kblob.e_ktype] << ". " << Flush;
    auto start = std::chrono::high_resolution_clock::now();
    oclr       = oclutil::cl_set_program(
      context, device_id, kblob.kernstr, sclp->clprog, build_opts, mowri, false);

    auto                          end   = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fp_ms = end - start;
    double                        secs  = fp_ms.count();
    std::string                   pre   = oclr.fail() ? "Failed in " : "Done in ";
    mowri << pre << std::setprecision(3) << secs << std::setprecision(6) << " [s]" << Endl;
  }
  return oclr;
}

void KernelTime::update_times(const cl_event& event)
{

  oclutil::cl_set_event_profiling_info(
    event, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start, nullptr, "u_times", true);

  oclutil::cl_set_event_profiling_info(
    event, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end, nullptr, "in update_times", true);

  auto new_time = 1e-6 * (t_end - t_start);
  v_times.push_back(new_time);
}

void KernelTime::reset_times()
{
  t_start = 0;
  t_end   = 0;
  v_times.resize(0);
}

void KernelTimes::reset_times()
{
  for (size_t pi = 0; pi < KType::E::N; ++pi)
  {
    ktimes[pi].reset_times();
  }
}

Programs::Programs(const cl_device_id& id, const cl_context& ctxt, owrite::Writer& mowri_)
  : act_inds(0), v_wait_indices(0), ptr_mowri(&mowri_)
{
  for (size_t pi = 0; pi < KType::E::N; ++pi)
  {
    programs[pi] = Program(id, ctxt);
  }
}

oclutil::Result Programs::update(const std::vector<KernBlob>& kbs)
{

  std::vector<std::string> warnings_to_ignore = {
    "conversion", "unused-macros", "shorten-64-to-32", "cast-align"};

  std::stringstream ss_build_options;
#ifndef __APPLE__
  //ss_build_options << "-Werror";
  ss_build_options << "   -cl-std=CL2.0";  // TODO : macro this.
  ss_build_options << "   -Wf,-Weverything";
  for (auto& x : warnings_to_ignore)
  {
    ss_build_options << "   -Wf,-Wno-" << x;
  }
#endif
  std::string build_options = ss_build_options.str();

  v_wait_indices = kerngen::get_v_wait_indices(kbs, *ptr_mowri);
  act_inds.resize(0);
  for (size_t kbi = 0; kbi < kbs.size(); ++kbi)
  {
    auto x = programs.at(kbs[kbi].e_ktype).update(kbs[kbi], *ptr_mowri, build_options);

    if (x.fail())
    {
      std::stringstream errm;
      errm << "failed to compile kernel in Programs::update : \n" << x.message;
      throw miog_error(errm.str());
    }

    act_inds.push_back(kbs[kbi].e_ktype);
  }
  return {};
}

oclutil::Result Programs::run(const cl_command_queue& queue,
                              const AllKernArgs&      all_args,
                              cl_uint                 n_user_wait_list,
                              const cl_event*         user_wait_list,
                              KernelTimes*            ptr_ktimes,
                              cl_event*               ptr_user_event,
                              bool                    debug_mode) const
{
  const bool             ev_from_user = (ptr_user_event != nullptr);
  auto                   n_active     = act_inds.size();
  std::vector<cl_kernel> clkerns(n_active);

  for (int k_ind = 0; k_ind < n_active; ++k_ind)
  {
    const Program& prog = programs[act_inds[k_ind]];
    ///////////////////////
    // Create the kernel //
    ///////////////////////
    if (debug_mode)
    {
      oclutil::cl_create_kernel(
        clkerns[k_ind], prog.sclp->clprog, prog.kblob.fname.c_str(), "programs run", true);
      oclutil::cl_set_kernel_args(clkerns[k_ind], all_args[k_ind], "programs run", true);
    }
    else
    {
      cl_int errcode;
      clkerns[k_ind] = clCreateKernel(prog.sclp->clprog, prog.kblob.fname.c_str(), &errcode);
      for (cl_uint arg_index = 0; arg_index < all_args[k_ind].size(); ++arg_index)
      {
        size_t      arg_size  = all_args[k_ind][arg_index].first;
        const void* arg_value = all_args[k_ind][arg_index].second;
        clSetKernelArg(clkerns[k_ind], arg_index, arg_size, arg_value);
      }
    }
  }

  std::vector<cl_event>  events(n_active - 1);
  std::vector<cl_event*> ptrs_events(n_active - 1);

  for (size_t i = 0; i < n_active - 1; ++i)
  {
    ptrs_events[i] = &events[i];
  }

  ptrs_events.emplace_back(ptr_user_event);

  for (int k_ind = 0; k_ind < n_active; ++k_ind)
  {
    const KernBlob& kblob = programs[act_inds[k_ind]].kblob;

    std::vector<cl_event> wait_list;
    for (cl_uint uw_ind = 0; uw_ind < n_user_wait_list; ++uw_ind)
    {
      wait_list.emplace_back(user_wait_list[uw_ind]);
    }
    for (auto& vw_ind : v_wait_indices[k_ind])
    {
      wait_list.emplace_back(*ptrs_events[vw_ind]);
    }
    const cl_event* ptr_wait_list = wait_list.size() == 0 ? nullptr : wait_list.data();

    ////////////////////////
    // Enqueue the kernel //
    ////////////////////////
    if (debug_mode)
    {

      if (!ev_from_user && ptr_ktimes != nullptr)
      {
        throw miog_error(
          "ktimes is not nullptr, and ev_from_user is false (ptr_user_event == nullptr)");
      }

      auto oclr = oclutil::cl_enqueue_ndrange_kernel(queue,
                                                     clkerns[k_ind],
                                                     1,
                                                     nullptr,
                                                     &kblob.global_work_size,
                                                     &kblob.local_work_size,
                                                     wait_list.size(),
                                                     ptr_wait_list,
                                                     ptrs_events[k_ind],
                                                     "run_kernels",
                                                     true);
    }
    else
    {

      clEnqueueNDRangeKernel(queue,
                             clkerns[k_ind],
                             1,
                             nullptr,
                             &kblob.global_work_size,
                             &kblob.local_work_size,
                             wait_list.size(),
                             ptr_wait_list,
                             ptrs_events[k_ind]);
    }
  }

  if (ev_from_user && ptr_ktimes != nullptr)
  {
    size_t maxend   = 0;
    size_t minstart = std::numeric_limits<size_t>::max();

    oclutil::cl_wait_for_events(1, ptrs_events.back(), "run742", true);
    for (int k_ind = 0; k_ind < n_active; ++k_ind)
    {
      KernelTime& pt = ptr_ktimes->ktimes[act_inds[k_ind]];
      pt.update_times(*ptrs_events[k_ind]);
      maxend   = std::max<size_t>(maxend, pt.t_end);
      minstart = std::min<size_t>(minstart, pt.t_start);
    }
    ptr_ktimes->extime = (1e-6 * (maxend - minstart));
  }

  if (debug_mode)
  {
    for (int k_ind = 0; k_ind < n_active - 1; ++k_ind)
    {
      oclutil::cl_release_event(events[k_ind], "event release", true);
    }

    for (int k_ind = 0; k_ind < n_active; ++k_ind)
    {
      oclutil::cl_release_kernel(clkerns[k_ind], "run" + std::to_string(k_ind), true);
    }
  }

  else
  {
    for (int k_ind = 0; k_ind < n_active - 1; ++k_ind)
    {
      clReleaseEvent(events[k_ind]);
    }

    for (int k_ind = 0; k_ind < n_active; ++k_ind)
    {
      clReleaseKernel(clkerns[k_ind]);
    }
  }

  return {};
}
}
