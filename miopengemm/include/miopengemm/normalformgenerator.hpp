/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_NORMALFORMGENERATOR_HPP
#define GUARD_MIOPENGEMM_NORMALFORMGENERATOR_HPP

#include <miopengemm/derivedparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/prepgenerator.hpp>

namespace MIOpenGEMM
{
namespace nformgen
{

KernelString get_nforma_kernelstring(const hyperparams::HyperParams&     hp,
                                     const Geometry&                     gg,
                                     const derivedparams::DerivedParams& dp);

KernelString get_nformb_kernelstring(const hyperparams::HyperParams&     hp,
                                     const Geometry&                     gg,
                                     const derivedparams::DerivedParams& dp);
}
}

#endif
