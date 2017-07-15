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

  //size_t              uninitialised_value{std::numeric_limits<size_t>::max()};

bool KernUses::uses(Mem::E emat_x) const  //(char c) const
{
  
  switch (emat_x){
    
    case Mem::E::A : return uses_a;
    case Mem::E::B : return uses_b;
    case Mem::E::C : return uses_c;
    case Mem::E::W : return uses_workspace;
    default : throw miog_error(std::string("unrecognised Mem::E in uses(.), ") + Mem::M.name[emat_x]);

  }
}

KernUses::KernUses(
  bool uses_a_, bool uses_b_, bool uses_c_, bool uses_workspace_, bool uses_alpha_, bool uses_beta_)
  : uses_a(uses_a_),
    uses_b(uses_b_),
    uses_c(uses_c_),
    uses_workspace(uses_workspace_),
    uses_alpha(uses_alpha_),
    uses_beta(uses_beta_)
{
  for (auto& x : {Mem::E::A, Mem::E::B, Mem::E::C, Mem::E::W})
  {
    if (uses(x))
    {
      full += Mem::M.name[x];
    }
  }

  if (uses_alpha)
  {
    full += "_alpha";
  }

  if (uses_beta)
  {
    full += "_beta";
  }

  //  we assume here that the main kernel will always use alpha
  if (uses_alpha)
  {
    e_kerntype = KType::MAIN;
  }

  else if (uses_beta && uses(Mem::E::C))
  {
    e_kerntype = KType::BETAC;
  }

  else if (uses(Mem::E::A) && uses(Mem::E::W))
  {
    e_kerntype = KType::WSA;
  }

  else if (uses(Mem::E::B) && uses(Mem::E::W))
  {
    e_kerntype = KType::WSB;
  }

  else
  {
    throw miog_error("determining `basic' string of KernUses, not sure what "
                     "this kernel does. "
                     "Its full string is " +
                     full);
  }
}
}
