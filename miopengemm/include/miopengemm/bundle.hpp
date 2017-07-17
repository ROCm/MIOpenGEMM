/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNELSTRINGGENERATOR_HPP
#define GUARD_MIOPENGEMM_KERNELSTRINGGENERATOR_HPP

#include <string>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM
{

namespace kerngen
{

class Bundle
{
  public:

  HyPas hp;
  Geometry gg;
  
  DerivedParams dp;
  std::vector<KernBlob>        v_tgks;
  std::vector<std::vector<size_t>> v_wait_indices;


  
  Bundle(const HyPas& hp, const Geometry& gg, owrite::Writer& mowri);
  
};

//Bundle get_bundle(const HyPas& hp, const Geometry& gg, owrite::Writer& mowri);
}
}

#endif
