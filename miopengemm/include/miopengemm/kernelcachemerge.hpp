/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNELCACHEMERGE_HPP
#define GUARD_MIOPENGEMM_KERNELCACHEMERGE_HPP



#include <miopengemm/kernelcache.hpp>

namespace MIOpenGEMM{
  

KernelCache get_merged(const KernelCache & kc1, const KernelCache & kc2);


}


#endif
