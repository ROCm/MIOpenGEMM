/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <limits>
#include <miopengemm/enums.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM
{

bool KernUses::at(Mem::E emat_x) const
{

  switch (emat_x)
  {

  case Mem::E::A: return u_a;
  case Mem::E::B: return u_b;
  case Mem::E::C: return u_c;
  case Mem::E::W: return u_w;
  case Mem::E::N: throw miog_error("N not allowed in KernUses::at");
  }
  throw miog_error("failed in KernUses::at");
}

KernUses::KernUses(bool u_a_, bool u_b_, bool u_c_, bool u_w_, bool u_alpha_, bool u_beta_)
  : u_a(u_a_), u_b(u_b_), u_c(u_c_), u_w(u_w_), u_alpha(u_alpha_), u_beta(u_beta_)
{
  for (auto& x : {Mem::E::A, Mem::E::B, Mem::E::C, Mem::E::W})
  {
    if (at(x))
    {
      full += Mem::M().name[x];
    }
  }

  if (u_alpha)
  {
    full += "_alpha";
  }

  if (u_beta)
  {
    full += "_beta";
  }
}
}
