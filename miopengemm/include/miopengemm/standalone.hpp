/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>

namespace MIOpenGEMM
{
namespace standalone
{

std::string make(const Geometry& gg, const HyPas& hp, owrite::Writer& mowri);

std::string reduce(const std::string& source);
}
}
