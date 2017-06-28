/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_BASICFIND_HPP
#define GUARD_MIOPENGEMM_BASICFIND_HPP

#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{

Solution basicfind(const Geometry&   geometry,
                   const Offsets&    toff,
                   const FindParams& find_params,
                   bool              verbose,
                   std::string       logfile,
                   std::string       constraints_string,
                   unsigned          n_postfind_runs,
                   bool              do_cpu_test);
}

#endif
