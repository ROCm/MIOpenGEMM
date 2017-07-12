/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <miopengemm/macgrid.hpp>
#include <sstream>
#include <cmath>

namespace MIOpenGEMM
{

namespace macgrid{

std::tuple<bool, std::string, std::array<size_t, 2>> get_grid(size_t mac, size_t skew)
{

  double   dbl_lg2_mac = std::log2(static_cast<double>(mac));
  size_t lg2_mac     = static_cast<size_t>(dbl_lg2_mac);

  double na = std::exp2(lg2_mac / 2 + lg2_mac % 2);
  double nb = static_cast<double>(mac) / na;
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

  std::stringstream errm_ss;

  errm_ss << "problem getting mac sizes: ";
  std::array<size_t, 2> null_array = {0, 0};
  size_t u_na = static_cast<size_t>(na);
  size_t u_nb = static_cast<size_t>(nb);
  if (std::abs(na * nb - static_cast<double>(u_na * u_nb)) > 1e-7)
  {
    errm_ss << "  casting non-ints. ";
    errm_ss << "na: " << na << " nb:" << nb << " u_na:" << u_na << " u_nb:" << u_nb << "  ";
    return std::make_tuple(false, errm_ss.str(), null_array);
  }

  if (u_na < 1 || u_nb < 1)
  {
    errm_ss << "  it appears that one of the lengths is zero. It could be that "
               "the skewness "
               "requested is too extreme. ";
    return std::make_tuple(false, errm_ss.str(), null_array);
  }

  if (u_na * u_nb != mac)
  {
    errm_ss << "  it appears as though the product of the computed edge "
               "lengths is not MAC.  ";
    return std::make_tuple(false, errm_ss.str(), null_array);
  }

  std::array<size_t, 2> mac_grid;

  if (Mat::E::A >= 2 || Mat::E::B >= 2)
  {
    errm_ss << "the std::array returned in get grid is too small";
    throw miog_error(errm_ss.str());
  }

  mac_grid[Mat::E::A] = u_na;
  mac_grid[Mat::E::B] = u_nb;

  return std::make_tuple(true, "no error", mac_grid);
}

}
}
