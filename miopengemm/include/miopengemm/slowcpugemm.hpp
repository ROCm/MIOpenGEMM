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
namespace slowcpugemm
{

template <typename TFloat>
void gemms_cpu(Geometry                     gg,
               Offsets                      toff,
               const TFloat*                a,
               const TFloat*                b,
               TFloat*                      c,
               TFloat                       alpha,
               TFloat                       beta,
               std::vector<std::string>     algs,
               owrite::Writer& mowri);

}
}

#endif
