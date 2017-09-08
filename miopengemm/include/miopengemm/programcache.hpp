/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_PROGRAMCACHE_HPP
#define GUARD_MIOPENGEMM_PROGRAMCACHE_HPP

#include <miopengemm/bundle.hpp>
//#include <miopengemm/generic.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernel.hpp>

namespace MIOpenGEMM
{

/*! @brief
 *  Encapsulation of all objects required to run GEMM for a particular geometry  */
class GemmKernelSquad
{

  public:
  std::vector<KernBlob>            v_kblobs;
  std::vector<Kernel>              kernels;
  std::vector<Kernel*>             ptr_kernels;
  std::vector<std::vector<size_t>> v_wait_indices;

  GemmKernelSquad() = default;

  GemmKernelSquad(const std::vector<KernBlob>& v_kblobs_,
                  owrite::Writer&              mowri,
                  cl_context                   context,
                  cl_device_id                 device_id);

  void set_args(const std::array<cl_mem, Mem::E::N>& gpu_mems,
                std::array<size_t, Mem::E::N>&       offsets,
                const void* vp_alpha,
                const void* vp_beta,
                size_t      floatsize_bytes);

  /*! clear all member vectors, effectively destroying object */
  void clear_vectors();
  
  void refresh_kernels();
};

template <typename T>
std::string adder(std::stringstream& ss, T t)
{
  ss << t;
  return ss.str();
}

template <typename T, typename... Args>
std::string adder(std::stringstream& ss, T first, Args... args)
{
  ss << first;
  return adder(ss, args...);
}

template <typename... Args>
std::string get_gemm_key(Args... args)
{
  std::stringstream ss;
  return adder(ss, args...);
}
}

#endif
