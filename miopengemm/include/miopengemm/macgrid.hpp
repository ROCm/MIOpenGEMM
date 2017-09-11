/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_SCOTMACGRID_HPP
#define GUARD_MIOPENGEMM_SCOTMACGRID_HPP

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
// at skew = skew0,
// MAC in (64, 256, ...)
// have a:b = 1:1
// and MAC in (32, 128,...)
// have a:b = 2:1
// at skew = skew0, (64, 256) are a:b = 1:1 and (32, 128) have a:b = 2:1
// at skew = skew0 + 1, (64, 256) are a:b = 1:4 and (32, 128) have a:b = 1:2
// at skew0 = skew + 1, (64, 256) are a:b = 4:1 and (32, 128) have a:b = 8:1
const size_t skew0 = 10;

// return true if power of 4.
bool mac_is_square(size_t mac);

class Grid
{

  private:
  void bad_initialise(const std::string& x);
  void good_initialise(size_t gA, size_t gB);
  size_t grid_A;
  size_t grid_B;

  public:
  bool        is_good;
  std::string error_message;
  Grid(size_t mac, size_t skew);
  size_t at(Mat::E emat);
};
}
}

#endif
