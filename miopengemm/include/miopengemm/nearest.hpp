/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_NEAREST_HPP
#define GUARD_MIOPENGEMM_NEAREST_HPP

#include <miopengemm/graph.hpp>
#include <miopengemm/kernelcache.hpp>

namespace MIOpenGEMM
{
namespace nearest
{

// for all CacheKeys, ck, in the KernelCache, which have
// (1) at(ck) with ck_in.gg is derivable.
// (2) at(ck) is in graph,
// is the {rank} closest to ck_in within radius threshold? rank = 0 for closest
bool is_within(
  const CacheKey& ck_in, const Graph&, const KernelCache&, double threshold, size_t rank);

// of all the CacheKeys in the KernelCache, return the {rank} nearest satisfying (1) and (2) above.
CacheKey get(const CacheKey&, const Graph&, const KernelCache&, size_t rank);
}
}

#endif
