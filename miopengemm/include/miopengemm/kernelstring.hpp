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

class KernUses
{

  public:
  // summarises use of uses_a, uses_b, uses_c
  std::string full;

  KType::E e_kerntype;

  bool uses_a;
  bool uses_b;
  bool uses_c;
  bool uses_workspace;
  bool uses_alpha;
  bool uses_beta;

  bool uses(Mem::E emat_x) const;

  KernUses(bool uses_a_,
             bool uses_b_,
             bool uses_c_,
             bool uses_workspace_,
             bool uses_alpha_,
             bool uses_beta_);

  KernUses() = default;
};

// TODO : this is a bad class name, as this is more than a *string*. change and propogate.
class KernBlobg
{
  public:
  // type : betac_alphab, betac_workspace, etc
  KernUses  type;
  std::string kernstr;
  std::string fname;

  size_t global_work_size;
  size_t local_work_size;

  KernBlobg(const KernUses&  type_,
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

  KernBlobg() = default;
};
}

#endif
