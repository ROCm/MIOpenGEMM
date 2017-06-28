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

#ifndef GUARD_MIOPENGEMM_RANDOMUTIL_HPP
#define GUARD_MIOPENGEMM_RANDOMUTIL_HPP

#include <algorithm>
#include <random>
#include <miopengemm/error.hpp>

namespace MIOpenGEMM
{

class RandomUtil
{

  private:
  std::random_device                      rd;
  std::default_random_engine              gen;
  std::uniform_int_distribution<unsigned> unidis;

  public:
  RandomUtil();
  unsigned get_from_range(unsigned upper);
  template <typename T>
  void shuffle(unsigned start_index, unsigned end_index, T& t)
  {
    if (end_index > t.size() || start_index > end_index)
    {
      throw miog_error("problem in template function RandomUtil::shuffle");
    }
    std::shuffle(t.begin() + start_index, t.begin() + end_index, gen);
  }
};
}

#endif
