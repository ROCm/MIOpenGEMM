/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_GEOMETRIESSS_HPP
#include <algorithm>
#include <sstream>
#include <vector>
#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM
{

std::vector<Geometry> get_deepbench(std::vector<size_t> wSpaceSize);
std::vector<Geometry> get_squares(std::vector<size_t> wSpaceSize);
std::vector<Geometry> take_fives(std::vector<size_t> wSpaceSize);
const std::vector<Geometry>& get_conv_geometries();
}

#endif
