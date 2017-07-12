/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_SCOTMACGRID_HPP
#define GUARD_MIOPENGEMM_SCOTMACGRID_HPP

#include <array>
#include <array>
#include <functional>
#include <map>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/oclutil.hpp>

namespace MIOpenGEMM
{
namespace macgrid
{

// macro tile shape
// at skew = skew0, MAC in (64, 256, ...)
// have a:b = 1:1 and MAC in (32, 128,...)
// have a:b = 2:1
// at skew = skew0, (64, 256) are a:b = 1:1 and (32, 128) have a:b = 2:1
// at skew = skew0 + 1, (64, 256) are a:b = 1:4 and (32, 128) have a:b = 1:2
// at skew0 = skew + 1, (64, 256) are a:b = 4:1 and (32, 128) have a:b = 8:1
const size_t skew0 = 10;

std::tuple<bool, std::string, std::array<size_t, 2>> get_grid(size_t mac, size_t skew);

}
}


#endif
