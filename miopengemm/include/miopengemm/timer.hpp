/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_TIMER_HPP
#define GUARD_MIOPENGEMM_TIMER_HPP

#include <chrono>

namespace MIOpenGEMM
{

// A stop-watch (with only a start button)
class Timer
{

  private:
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;

  public:
  void   start();
  double get_elapsed() const;
};
}

#endif
