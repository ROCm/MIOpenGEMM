/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNELCACHEMERGE_HPP
#define GUARD_MIOPENGEMM_KERNELCACHEMERGE_HPP

#include <miopengemm/findparams.hpp>
#include <miopengemm/kernelcache.hpp>

namespace MIOpenGEMM
{
KernelCache
get_merged(const KernelCache& kc1, const KernelCache& kc2, const Halt& halt, owrite::Writer& mowri);

KernelCache get_wSpaceReduced(const KernelCache& kc);
}

#endif
