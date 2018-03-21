/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <sstream>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/tiling.hpp>
#include <cassert>

namespace MIOpenGEMM
{
namespace tiling
{

std::vector<size_t> get_multiples(size_t N)
{
  std::vector<size_t> multiples;
  for (size_t k = N; k > 0; --k)
  {
    if (N % k == 0)
    {
      multiples.push_back(k);
    }
  }
  return multiples;
}

void set_tile_dimensions_no_checks(size_t& tH, size_t& tW, size_t TH, size_t TW, size_t tS)
{
  for (auto& multiple_of_TH : get_multiples(TH))
  {
    if ((tS % multiple_of_TH == 0) && ((tS / multiple_of_TH) <= TW))
    {
      tH = multiple_of_TH;
      tW = tS / tH;
      break;
    }
  }
}

std::tuple<bool, std::string> get_tileability(size_t TH, size_t TW, size_t tS)
{
  std::stringstream ss_tileable_status;

  if (tS == 0)
  {
    std::stringstream errm;
    errm << "In get_tileability, and tS is zero. "
         << "This is worse than non-tileable, "
         << "there is probably a bad input parameter.";

    throw miog_error(errm.str());
  }

  std::string       set_ds("");
  std::stringstream input_ss;
  input_ss << '\n' << "TH : " << TH << " TW : " << TW << " tS : " << tS;
  std::string input_string = input_ss.str();

  if ((TH * TW) % tS != 0)
  {
    ss_tileable_status << "Areas of micro and macro tiles are incompatible : " << input_string;
    std::make_tuple(false, ss_tileable_status.str());
  }

  size_t tH = 0;
  size_t tW = 0;
  set_tile_dimensions_no_checks(tH, tW, TH, TW, tS);

  if (tH == 0)
  {
    ss_tileable_status << "Impossible tiling problem in get_tile_dimensions : " << input_string;
    std::make_tuple(false, ss_tileable_status.str());
  }

  if (tW > tH)
  {
    // this would be a pedantic error: no `tall' tile. best `wide' one " + bla
  }

  assert(tW != 0); 

  if (TW % tW != 0 || TH % TH != 0 || tW * tH != tS)
  {
    std::stringstream err_ss;
    err_ss << "Problem in get_tileability."
           << " This isn't even non-tileable, this is a logic error. "
           << "The found micro tile size is not consistent with the macro tile : " << input_string
           << "   tH : " << tH << " tW  " << tW;
    throw miog_error(err_ss.str());
  }

  // ran the gauntlet successfully
  return std::make_tuple(true, "");
}

void set_tile_dimensions(size_t& tH, size_t& tW, size_t TH, size_t TW, size_t tS, bool tall)
{

  bool        is_tileable;
  std::string tileable_status;
  std::tie(is_tileable, tileable_status) = get_tileability(TH, TW, tS);

  if (is_tileable == false)
  {
    std::stringstream errm;
    errm << "In set_tile_dimensions, and the problem is not tileable."
         << " Call get_tileability as a check before set_tile_dimensions to catch this case "
         << "without throwing an error. "
         << "The string returned from set_tile_dimensions was : " << tileable_status;

    throw miog_error(errm.str());
  }

  // switch (tW <-> tH) and (TW <-> TH)
  if (tall == false)
  {
    set_tile_dimensions_no_checks(tW, tH, TW, TH, tS);
  }

  else
  {
    set_tile_dimensions_no_checks(tH, tW, TH, TW, tS);
  }
}
}
}
