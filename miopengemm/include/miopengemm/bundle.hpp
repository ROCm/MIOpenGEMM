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
  const std::vector<KernBlobg>        v_tgks;
  const std::vector<std::vector<size_t>> v_wait_indices;

  DerivedParams dp;

  Bundle(std::vector<KernBlobg>&&        v_tgks_,
         std::vector<std::vector<size_t>>&& v_wait_indices_,
         DerivedParams&&                    dp_)
    : v_tgks(std::move(v_tgks_)), v_wait_indices(std::move(v_wait_indices_)), dp(std::move(dp_))
  {
  }
};

Bundle get_bundle(const HyPas& hp, const Geometry& gg, owrite::Writer& mowri);
}
}

#endif
