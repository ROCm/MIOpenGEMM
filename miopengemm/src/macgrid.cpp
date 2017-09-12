/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <cmath>
#include <sstream>
#include <miopengemm/macgrid.hpp>

namespace MIOpenGEMM
{

namespace macgrid
{

bool mac_is_square(size_t mac)
{
  // if not a power of 2, return false
  if (mac & (mac - 1))
  {
    return false;
  }
  // x & 10101010101010
  return mac & 0x55555555;
}

void Grid::bad_initialise(const std::string& x)
{
  is_good       = false;
  error_message = x;
}

void Grid::good_initialise(size_t gA, size_t gB)
{
  grid_A        = gA;
  grid_B        = gB;
  is_good       = true;
  error_message = "";
}

Grid::Grid(size_t mac, size_t skew)  // 32, 9
{

  double dbl_lg2_mac = std::log2(static_cast<double>(mac));  // 5
  size_t lg2_mac     = static_cast<size_t>(dbl_lg2_mac);     // 5

  double na = std::exp2(lg2_mac / 2 + lg2_mac % 2);  // 8
  double nb = static_cast<double>(mac) / na;         // 4
  for (size_t i = skew0; i < skew; ++i)
  {
    na /= 2.;
    nb *= 2.;
  }

  for (size_t i = skew; i < skew0; ++i)
  {
    na *= 2.;
    nb /= 2.;
  }

  size_t u_na = static_cast<size_t>(na);
  size_t u_nb = static_cast<size_t>(nb);
  if (std::abs(na * nb - static_cast<double>(u_na * u_nb)) > 1e-7)
  {
    std::stringstream errm_ss;
    errm_ss << "Casting non-ints. ";
    errm_ss << "na: " << na << " nb:" << nb << " u_na:" << u_na << " u_nb:" << u_nb << '.';
    bad_initialise(errm_ss.str());
    return;
  }

  if (u_na < 1 || u_nb < 1)
  {
    bad_initialise("One of the lengths is zero. Maybe skewness requested is too extreme.");
    return;
  }

  if (u_na * u_nb != mac)
  {
    bad_initialise("The product of the computed edge lengths is not MAC.");
    return;
  }

  good_initialise(u_na, u_nb);
}

size_t Grid::at(Mat::E emat)
{
  if (is_good == false)
  {
    throw miog_error("at should not be called as is_good is false, internal logic error");
  }

  if (emat == Mat::E::A)
  {
    return grid_A;
  }
  else if (emat == Mat::E::B)
  {
    return grid_B;
  }
  else
  {
    throw miog_error("unrecognised emat in Grid::at, internal logic error");
  }
}
}
}
