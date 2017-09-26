/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <miopengemm/bundle.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/programcacher.hpp>
#include <miopengemm/programs.hpp>
#include <miopengemm/timer.hpp>
#include <miopengemm/tinyzero.hpp>

namespace MIOpenGEMM
{

// TODO : alpha = 0 optimisation. beta = 0 optimisation.
template <typename T>
GemmStatus xgemm(bool              isColMajor,
                 bool              tA,
                 bool              tB,
                 size_t            m,
                 size_t            n,
                 size_t            k,
                 T                 alpha,
                 cl_mem            a,
                 size_t            a_offset,
                 size_t            lda,
                 cl_mem            b,
                 size_t            b_offset,
                 size_t            ldb,
                 T                 beta,
                 cl_mem            c,
                 size_t            c_offset,
                 size_t            ldc,
                 cl_mem            w,
                 size_t            w_offset,
                 size_t            w_size,
                 cl_command_queue* ptr_queue,
                 cl_uint           num_events_in_wait_list,
                 const cl_event*   event_wait_list,
                 cl_event*         ptr_event_user,
                 int               ID)
{

  if (ID < 0)
  {

    BetaType beta_type = get_beta_type(beta);

    ID = get_cacher().get_ID(isColMajor,
                             tA,
                             tB,
                             false,  // tC not passed to xgemm.
                             m,
                             n,
                             k,
                             lda,
                             ldb,
                             ldc,
                             w_size,
                             beta_type,
                             get_floattype_char<T>(),
                             ptr_queue);
  }

  const Programs& programs = get_cacher().program_cache[ID];

  std::array<cl_mem, Mem::E::N> gpu_mems;
  std::array<size_t, Mem::E::N> offsets;

  gpu_mems[Mem::E::A] = a;
  gpu_mems[Mem::E::B] = b;
  gpu_mems[Mem::E::C] = c;
  gpu_mems[Mem::E::W] = w;

  offsets[Mem::E::A] = a_offset;
  offsets[Mem::E::B] = b_offset;
  offsets[Mem::E::C] = c_offset;
  offsets[Mem::E::W] = w_offset;

  AllKernArgs all_kern_args(0);
  for (auto& index : programs.act_inds)
  {
    auto& program = programs.programs[index];
    all_kern_args.emplace_back(
      kerngen::get_arg_sizes_values(program.kblob, gpu_mems, offsets, sizeof(T), &alpha, &beta));
  }

  KernelTimes* ktimes     = nullptr;
  bool         debug_mode = false;
  programs.run(*ptr_queue,
               all_kern_args,
               num_events_in_wait_list,
               event_wait_list,
               ktimes,  // update_times,
               ptr_event_user,
               debug_mode);

  return {true, ID};
}

template GemmStatus xgemm<float>(bool,
                                 bool,
                                 bool,
                                 size_t,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_command_queue*,
                                 cl_uint,
                                 const cl_event*,
                                 cl_event*,
                                 int ID);

template GemmStatus xgemm<double>(bool,
                                  bool,
                                  bool,
                                  size_t,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_command_queue*,
                                  cl_uint,
                                  const cl_event*,
                                  cl_event*,
                                  int ID);

// TODO : beta = 1 optimisation. alpha = 0 optimisation. beta = 0 optimisation.
template <typename T>
GemmStatus gemm0(bool              isColMajor,
                 bool              tA,
                 bool              tB,
                 size_t            m,
                 size_t            n,
                 size_t            k,
                 T                 alpha,
                 cl_mem            a,
                 size_t            a_offset,
                 size_t            lda,
                 cl_mem            b,
                 size_t            b_offset,
                 size_t            ldb,
                 T                 beta,
                 cl_mem            c,
                 size_t            c_offset,
                 size_t            ldc,
                 cl_command_queue* ptr_queue,
                 cl_uint           num_events_in_wait_list,
                 const cl_event*   event_wait_list,
                 cl_event*         ptr_event_user)
{
  return xgemm<T>(isColMajor,
                  tA,
                  tB,
                  m,
                  n,
                  k,
                  alpha,
                  a,
                  a_offset,
                  lda,
                  b,
                  b_offset,
                  ldb,
                  beta,
                  c,
                  c_offset,
                  ldc,
                  nullptr,
                  0,
                  0,
                  ptr_queue,
                  num_events_in_wait_list,
                  event_wait_list,
                  ptr_event_user,
                  -1);
}

template GemmStatus gemm0<float>(bool,
                                 bool,
                                 bool,
                                 size_t,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_command_queue*,
                                 cl_uint,
                                 const cl_event*,
                                 cl_event*);

template GemmStatus gemm0<double>(bool,
                                  bool,
                                  bool,
                                  size_t,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_command_queue*,
                                  cl_uint,
                                  const cl_event*,
                                  cl_event*);
}
