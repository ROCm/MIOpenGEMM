/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <limits>
#include <sstream>
#include <miopengemm/enums.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM
{

bool KernUses::at(Mat::E emat_x) const
{

  switch (emat_x)
  {

  case Mat::E::A: return u_a;
  case Mat::E::B: return u_b;
  case Mat::E::C: return u_c;
  case Mat::E::N: throw miog_error("N not allowed in KernUses::at");
  }
  throw miog_error("failed in KernUses::at");
}

KernUses::KernUses(
  bool u_a_, bool u_b_, bool u_c_, std::vector<bool> u_vws_, bool u_alpha_, bool u_beta_, bool u_k_)
  : u_a(u_a_), u_b(u_b_), u_c(u_c_), u_vws(u_vws_), u_alpha(u_alpha_), u_beta(u_beta_), u_k(u_k_)
{
  
  std::stringstream fullss;
  for (auto& x : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    if (at(x))
    {
      if (!fullss.str().empty()){
        fullss << '_';
      }
      fullss << Mat::M().name[x];
    }
  }

  if (u_vws.size() != 0)
  {
    for (size_t i = 0; i < u_vws.size(); ++i)
    {
      if (u_vws[i])
      {
      if (!fullss.str().empty()){
        fullss << '_';
      }        
      fullss <<  'W'  << i;
      }
    }
  }

  if (u_alpha)
  {
    if (!fullss.str().empty()){
      fullss << '_';
    }            
    fullss << "alpha";
  }

  if (u_beta)
  {
    if (!fullss.str().empty()){
      fullss << '_';
    }            
    fullss <<  "beta";
  }
  full = fullss.str();
}
}
