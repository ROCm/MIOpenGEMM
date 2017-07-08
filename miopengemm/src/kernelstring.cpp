/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <iostream>
#include <limits>
#include <miopengemm/error.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/enums.hpp>

namespace MIOpenGEMM
{


std::vector<std::vector<unsigned>> get_kernel_dependencies()
{

  unsigned              uninitialised_value{std::numeric_limits<unsigned>::max()};
  std::vector<unsigned> uninitialised_vector{uninitialised_value};

  std::vector<std::vector<unsigned>> kdps(BasicKernelType::N, uninitialised_vector);

  kdps[BasicKernelType::WSA]   = {};
  kdps[BasicKernelType::WSB]   = {};
  kdps[BasicKernelType::BETAC] = {};
  kdps[BasicKernelType::MAIN]  = {BasicKernelType::BETAC, BasicKernelType::WSA, BasicKernelType::WSB};

  for (unsigned i = 0; i < BasicKernelType::N; ++i)
  {
    if (kdps[i].size() == 1 && kdps[i][0] == uninitialised_value)
    {
      throw miog_error("kernel_dependencies does not appear to be initialised entirely");
    }
  }

  return kdps;
}

//const std::vector<std::string> basic_kernel_type_strings     = get_basic_kernel_type_strings();
const std::vector<std::vector<unsigned>> kernel_dependencies = get_kernel_dependencies();

bool KernelType::uses(char c) const
{
  if (c == 'a')
  {
    return uses_a;
  }
  else if (c == 'b')
  {
    return uses_b;
  }
  else if (c == 'c')
  {
    return uses_c;
  }
  else if (c == 'w')
  {
    return uses_workspace;
  }
  else
  {
    throw miog_error(std::string("unrecognised char in uses in KernelType, ") + c);
  }
}

KernelType::KernelType(
  bool uses_a_, bool uses_b_, bool uses_c_, bool uses_workspace_, bool uses_alpha_, bool uses_beta_)
  : uses_a(uses_a_),
    uses_b(uses_b_),
    uses_c(uses_c_),
    uses_workspace(uses_workspace_),
    uses_alpha(uses_alpha_),
    uses_beta(uses_beta_)
{
  for (auto& x : {'a', 'b', 'c', 'w'})
  {
    if (uses(x))
    {
      full += x;
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
    basic_kernel_type = BasicKernelType::MAIN;
    bkt_string        = BasicKernelType::M.name[BasicKernelType::MAIN];
  }

  else if (uses_beta && uses('c'))
  {
    basic_kernel_type = BasicKernelType::BETAC;
    bkt_string        = BasicKernelType::M.name[BasicKernelType::BETAC];
  }

  else if (uses('a') && uses('w'))
  {
    basic_kernel_type = BasicKernelType::WSA;
    bkt_string        = BasicKernelType::M.name[BasicKernelType::WSA];
  }

  else if (uses('b') && uses('w'))
  {
    basic_kernel_type = BasicKernelType::WSB;
    bkt_string        = BasicKernelType::M.name[BasicKernelType::WSB];
  }

  else
  {
    throw miog_error("determining `basic' string of KernelType, not sure what "
                     "this kernel does. "
                     "Its full string is " +
                     full);
  }
}
}
