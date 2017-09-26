/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
  std::random_device                    rd;
  std::default_random_engine            gen;
  std::uniform_int_distribution<size_t> unidis;

  public:
  RandomUtil();
  RandomUtil(int seed);
  size_t get_from_range(size_t upper);
  template <typename T>
  void shuffle(size_t start_index, size_t end_index, T& t)
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
