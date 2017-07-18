/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_TILING_HPP
#define GUARD_MIOPENGEMM_TILING_HPP

#include <string>
#include <tuple>

namespace MIOpenGEMM
{
namespace tiling
{

// given a macro tile TH x TW,
// and given a micro tile size of tS,
// find the tallest (if tall = true, otherwise widest)
// possible micro tile size (tH x tW)
// to fit the macro tile. Example, macro tile is 6 x 4:
//* * * *
//* * * *
//* * * *
//* * * *
//* * * *
//* * * *
// tS = 2 return [2, 1]
// tS = 3 return [3, 1]
// tS = 4 return [2, 2]
// tS = 5 raise an error ((TH * TH) % tS != 0)
// tS = 6 return [6, 1]
// tS = 7 raise an error ((TH * TH) % tS != 0)
// tS = 8 return [2, 4]
// tS = 9 raise an error ((TH * TH) % tS != 0)
// tS = 10 raise an error ((TH * TH) % tS != 0)
// tS = 11 raise an error ((TH * TH) % tS != 0)
// tS = 12 return [6, 2]
// tS = 13 .. 23 raise an error ((TH * TH) % tS != 0)
// tS = 24 return [6, 4]

void set_tile_dimensions(size_t& tH, size_t& tW, size_t TH, size_t TW, size_t tS, bool tall = true);

// checks if it is tileable according to the above.
// returns (true, "") if, otherwise (false,"reason")
std::tuple<bool, std::string> get_tileability(size_t TH, size_t TW, size_t tS);
}
}

#endif
