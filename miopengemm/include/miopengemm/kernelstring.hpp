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
  // summarises use of u_a, u_b, u_c
  std::string full;

  bool u_a = false;
  bool u_b = false;
  bool u_c = false;
  bool u_w = false;
  bool u_alpha = false;
  bool u_beta = false;

  bool at(Mem::E emat_x) const;

  KernUses(bool u_a_, bool u_b_, bool u_c_, bool u_w_, bool u_alpha_, bool u_beta_);

  KernUses() = default;
};

class KernBlob
{
  public:
  KType::E    e_ktype = KType::E::N;
  KernUses    kuses;
  std::string kernstr;
  std::string fname;

  size_t global_work_size;
  size_t local_work_size;

  KernBlob(KType::E           e_ktype_,
           const KernUses&    kuses_,
           std::string&&      kernstr_,
           const std::string& fname_,
           size_t             global_work_size_,
           size_t             local_work_size_)
    : e_ktype(e_ktype_),
      kuses(kuses_),
      kernstr(kernstr_),
      fname(fname_),
      global_work_size(global_work_size_),
      local_work_size(local_work_size_)
  {
  }

  KernBlob() = default;
};
}

#endif
