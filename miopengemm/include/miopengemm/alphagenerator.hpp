
/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/derivedparams.hpp>

namespace MIOpenGEMM{
namespace alphagen{
  
KernelString get_alpha_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp);
 
}
}
