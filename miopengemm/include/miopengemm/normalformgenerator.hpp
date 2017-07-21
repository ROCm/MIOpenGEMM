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

KernBlob
get_nform_kernelstring(Mat::E emat_x, const HyPas& hp, const Geometry& gg, const DerivedParams& dp);
}
}

#endif
