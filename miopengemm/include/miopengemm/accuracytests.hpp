/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ACCURACYTESTS_HPP
#define GUARD_MIOPENGEMM_ACCURACYTESTS_HPP

#include <algorithm>
#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{
namespace accuracytests
{

template <typename TFloat>
void elementwise_compare(
  const Geometry& gg,
  const Offsets&  toff,
  const TFloat*   c_before,   // C matrix before GEMM
  const TFloat*   c_cpu,      // C matrix after GEMM on CPU
  const TFloat*   c_gpu,      // C matrix after GEMM on GPU.
  const TFloat*   c_cpu_abs,  // C matrix after GEMM : abs(alpha)*abs(A)abs(B) + abs(beta)*abs(C)
  std::string     info_str,   // to be printed in error message if there is a problem
  owrite::Writer& mowri);
}
}

#endif
