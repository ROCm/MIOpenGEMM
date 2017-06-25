
/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef SLOWCPUGEMM
#define SLOWCPUGEMM

#include <vector>
#include <string>

#include <miopengemm/outputwriter.hpp>
#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM{
namespace slowcpugemm{


template <typename TFloat>
void gemms_cpu(Geometry gg, Offsets toff, const TFloat * a, const TFloat * b, TFloat * c, TFloat alpha, TFloat beta, std::vector<std::string> algs, outputwriting::OutputWriter & mowri);
      
} //namespace
}


#endif
