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

Program::Program(cl_device_id id, cl_context ctxt) : device_id(id), context(ctxt), clprog(nullptr)
{
}

void Program::try_release()
{
  if (clprog != nullptr)
  {
    oclutil::cl_release_program(clprog, "try_release", true);
  }
}

oclutil::Result
Program::update(const KernBlob& ks, owrite::Writer& mowri, const std::string& build_opts)
{

  oclutil::Result oclr;

  if ((clprog != nullptr) && (ks.kernstr == kblob.kernstr))
  {
    oclr = {};
  }

  else
  {
    try_release();
    kblob = ks;
    mowri << "compiling " << KType::M.name[kblob.e_ktype] << ". " << Flush;
    auto start = std::chrono::high_resolution_clock::now();
    oclr =
      oclutil::cl_set_program(context, device_id, kblob.kernstr, clprog, build_opts, mowri, false);
    auto                         end   = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fp_ms = end - start;
    float                        secs  = fp_ms.count();
    std::string                  pre   = oclr.fail() ? "Failed in " : "Done in ";
    mowri << pre << std::setprecision(3) << secs << std::setprecision(6) << " [s]" << Endl;
  }
  return oclr;
}

Program::~Program() { try_release(); }

void Program::update_times(const cl_event& event)
{

  oclutil::cl_set_event_profiling_info(
    event, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start, nullptr, "u_times", true);

  oclutil::cl_set_event_profiling_info(
    event, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end, nullptr, "in update_times", true);

  v_times.push_back(1e-6 * (t_end - t_start));
}

void Program::reset_times()
{
  t_start = 0;
  t_end   = 0;
  v_times.resize(0);
}

BasePrograms::BasePrograms(const cl_device_id& id, const cl_context& ctxt, owrite::Writer& mowri_)
  : active_program_indices(0), v_wait_indices(0), ptr_mowri(&mowri_)
{
  for (size_t pi = 0; pi < KType::E::N; ++pi)
  {
    programs[pi] = Program(id, ctxt);
  }
}

void BasePrograms::update(const std::vector<KernBlob>& kbs)
{

  std::string build_options("-cl-std=CL2.0  -Werror");

  // v_wait_indices.
  v_wait_indices = kerngen::get_v_wait_indices(kbs, *ptr_mowri);
  // set active_program_indices.
  active_program_indices.resize(0);
  for (size_t kbi = 0; kbi < kbs.size(); ++kbi)
  {
    // set programs.
    programs.at(kbs[kbi].e_ktype).update(kbs[kbi], *ptr_mowri, build_options);
    active_program_indices.push_back(kbs[kbi].e_ktype);
  }
}

oclutil::Result VerbosePrograms::run(const cl_command_queue& queue,
                                     AllKernArgs&            all_args,
                                     cl_uint                 n_user_wait_list,
                                     const cl_event*         user_wait_list,
                                     bool                    update_times,
                                     cl_event*               ptr_user_event)
{
  // TODO RAII these.
  auto n_active = active_program_indices.size();

  std::vector<cl_event> clevents(n_active);
  if (ptr_user_event != nullptr)
  {
    clevents[n_active - 1] = *ptr_user_event;
  }

  std::vector<cl_kernel> clkerns(n_active);
  for (int k_ind = 0; k_ind < n_active; ++k_ind)
  {
    Program& prog = programs[active_program_indices[k_ind]];
    oclutil::cl_create_kernel(
      clkerns[k_ind], prog.clprog, prog.kblob.fname.c_str(), "setting kernel in ::run", true);
  }

  for (int k_ind = 0; k_ind < n_active; ++k_ind)
  {

    Program& prog = programs[active_program_indices[k_ind]];

    std::vector<cl_event> wait_list;

    for (cl_uint uw_ind = 0; uw_ind < n_user_wait_list; ++uw_ind)
    {
      wait_list.emplace_back(user_wait_list[uw_ind]);
    }

    for (auto& vw_ind : v_wait_indices[k_ind])
    {
      wait_list.emplace_back(clevents[vw_ind]);
    }

    const cl_event* ptr_wait_list = wait_list.size() == 0 ? nullptr : wait_list.data();

    oclutil::cl_set_kernel_args(clkerns[k_ind], all_args[k_ind], "Program::set_kernel_args", true);

    auto oclr = oclutil::cl_enqueue_ndrange_kernel(queue,
                                                   clkerns[k_ind],
                                                   1,
                                                   nullptr,
                                                   &prog.kblob.global_work_size,
                                                   &prog.kblob.local_work_size,
                                                   wait_list.size(),
                                                   wait_list.data(),
                                                   &clevents[k_ind],
                                                   "run_kernels",
                                                   false);
  }

  for (int k_ind = 0; k_ind < n_active; ++k_ind)
  {
    oclutil::cl_release_kernel(clkerns[k_ind], "release kernel " + std::to_string(k_ind), true);
  }

  for (int k_ind = 0; k_ind < n_active - 1; ++k_ind)
  {
    oclutil::cl_release_event(clevents[k_ind], "release cl_event " + std::to_string(k_ind), true);
  }

  if (ptr_user_event != nullptr)
  {
    oclutil::cl_release_event(clevents[n_active - 1], "release final cl_event", true);
  }
}
}
