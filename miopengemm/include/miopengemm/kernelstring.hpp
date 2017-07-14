/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNELSTRINGS_HPP
#define GUARD_MIOPENGEMM_KERNELSTRINGS_HPP

#include <string>
#include <vector>
#include <miopengemm/enums.hpp>

namespace MIOpenGEMM
{

// maps dependencies of execution order of kernels
extern const std::vector<std::vector<size_t>> kernel_dependencies;

class KernelType
{

  public:
  // summary of uses_a, uses_b, uses_c etc
  std::string full;

  // one of wsa, wsb, betac, main
  std::string bkt_string;

  BasicKernelType::E basic_kernel_type;

  bool uses_a;
  bool uses_b;
  bool uses_c;
  bool uses_workspace;
  bool uses_alpha;
  bool uses_beta;

  bool uses(Mem::E emat_x) const;

  KernelType(bool uses_a_,
             bool uses_b_,
             bool uses_c_,
             bool uses_workspace_,
             bool uses_alpha_,
             bool uses_beta_);

  KernelType() = default;
};

// TODO : this is a bad class name, as this is more than a *string*. change and propogate.
class KernelString
{
  public:
  // type : betac_alphab, betac_workspace, etc
  KernelType  type;
  std::string kernstr;
  std::string fname;

  size_t global_work_size;
  size_t local_work_size;

  KernelString(const KernelType&  type_,
               std::string&&      kernstr_,
               const std::string& fname_,
               size_t             global_work_size_,
               size_t             local_work_size_)
    : type(type_),
      kernstr(kernstr_),
      fname(fname_),
      global_work_size(global_work_size_),
      local_work_size(local_work_size_)
  {
  }

  KernelString() = default;
};
}

#endif
