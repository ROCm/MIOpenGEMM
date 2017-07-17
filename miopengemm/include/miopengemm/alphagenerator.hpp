/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ALPHAGENERATOR_HPP
#define GUARD_MIOPENGEMM_ALPHAGENERATOR_HPP

#include <miopengemm/derivedparams.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM
{
namespace alphagen
{

KernBlob get_alpha_kernelstring(const HyPas& hp, const Geometry& gg, const DerivedParams& dp);
}
}

#endif
