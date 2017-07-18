/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <miopengemm/timer.hpp>

namespace MIOpenGEMM
{

void Timer::start() { t0 = std::chrono::high_resolution_clock::now(); }

double Timer::get_elapsed() const
{
  std::chrono::duration<double> fp_ms = std::chrono::high_resolution_clock::now() - t0;
  return fp_ms.count();
}
}
