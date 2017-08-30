/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <miopengemm/programcache.hpp>

namespace MIOpenGEMM
{

void GemmKernelSquad::clear_vectors(){
  v_kblobs.clear();
  kernels.clear();
  ptr_kernels.clear();
  v_wait_indices.clear();
}

GemmKernelSquad::GemmKernelSquad(const std::vector<KernBlob>& v_kblobs_,
                                 owrite::Writer&              mowri,
                                 cl_context                   context,
                                 cl_device_id                 device_id)
  : v_kblobs(v_kblobs_)
{
  size_t n_kernels = v_kblobs.size();
  kernels          = std::vector<Kernel>(n_kernels);
  ptr_kernels.resize(n_kernels);
  for (auto ksi = 0; ksi < n_kernels; ++ksi)
  {
    kernels[ksi] = Kernel(device_id, context, nullptr, KType::M.name[v_kblobs[ksi].e_ktype]);
    kernels[ksi].update(v_kblobs[ksi], mowri);
    ptr_kernels[ksi] = &kernels[ksi];
  }
  v_wait_indices = kerngen::get_v_wait_indices(v_kblobs, mowri);
}

void GemmKernelSquad::set_args(const std::array<cl_mem, Mem::E::N>& gpu_mems,
                               std::array<size_t, Mem::E::N>&       offsets,
                               const void* vp_alpha,
                               const void* vp_beta,
                               size_t      floatsize_bytes)
{
  for (auto ksi = 0; ksi < v_kblobs.size(); ++ksi)
  {
    auto arg_sizes_values = kerngen::get_arg_sizes_values(
      v_kblobs[ksi], gpu_mems, offsets, floatsize_bytes, vp_alpha, vp_beta);

    bool clearer_error = false;

    if (clearer_error)
    {
      kernels[ksi].set_kernel_args(arg_sizes_values);
    }

    else
    {
      for (size_t arg_index = 0; arg_index < arg_sizes_values.size(); ++arg_index)
      {
        auto   arg_size  = std::get<0>(arg_sizes_values[arg_index]);
        auto   arg_value = std::get<1>(arg_sizes_values[arg_index]);
        cl_int ret       = clSetKernelArg(kernels[ksi].clkern, arg_index, arg_size, arg_value);
        if (ret != CL_SUCCESS)
        {
          throw miog_error("failed to set arg" + std::to_string(arg_index) + " (programcache)");
        }
      }
    }
  }
}
}
