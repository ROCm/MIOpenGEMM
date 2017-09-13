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

// parameter order rule: {a, oa, b, ob, c, oc, ws, ows}, alpha, beta
std::vector<std::pair<size_t, const void*>>
get_arg_sizes_values(const KernBlob& kblob,
                     const std::array<cl_mem, Mem::E::N>& cl_mems,
                     const std::array<size_t, Mem::E::N>& offsets,
                     size_t      float_size_bytes,
                     const void* alpha,
                     const void* beta);

std::vector<std::vector<size_t>> get_v_wait_indices(const std::vector<KernBlob>& v_kblobs,
                                                    owrite::Writer&              mowri);

class Bundle
{
  public:
  HyPas    hp;
  Geometry gg;

  DerivedParams         dp;
  std::vector<KernBlob> v_tgks;

  Bundle(const HyPas& hp, const Geometry& gg);
};
}
}

#endif
