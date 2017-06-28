/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
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

void set_tile_dimensions(
  unsigned& tH, unsigned& tW, unsigned TH, unsigned TW, unsigned tS, bool tall = true);

// checks if it is tileable according to the above.
// returns (true, "") if, otherwise (false,"reason")
std::tuple<bool, std::string> get_tileability(unsigned TH, unsigned TW, unsigned tS);
}
}

#endif
