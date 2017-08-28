/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_SLOWCPUGEMM_HPP
#define GUARD_MIOPENGEMM_SLOWCPUGEMM_HPP

#include <string>
#include <vector>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{
namespace cpugemm
{

template <typename TFloat>
void gemm(Geometry        gg,
          Offsets         toff,
          const TFloat*   a,
          const TFloat*   b,
          TFloat*         c,
          TFloat          alpha,
          TFloat          beta,
          owrite::Writer& mowri);
}
}

#endif
