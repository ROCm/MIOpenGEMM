/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_GENERIC_HPP
#define GUARD_MIOPENGEMM_GENERIC_HPP

#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/solution.hpp>

// no kernels are run with these methods, they rely only on cache
namespace MIOpenGEMM
{

// not even a cache look-up, just return 1 of 3 non-optimsed kernels
HyPas get_generic(const Geometry& gg, const Constraints& constraints);

// try and get a solution from cache, if all else fails get_generic.
Solution get_default(cl_command_queue   command_queue,
                     const Geometry&    gg,
                     const Constraints& constraints,
                     owrite::Writer&    mowri,
                     IfNoCache::E       enoc, 
                     size_t rank);
}

#endif
