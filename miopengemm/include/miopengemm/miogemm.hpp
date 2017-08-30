/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_MIOGEMM_HPP
#define GUARD_MIOPENGEMM_MIOGEMM_HPP

#include <CL/cl.h>
#include <miopengemm/findparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{

/*! @brief
 * Find a Solution, considering only Solutions which do not use workspace.
 * This function is deprecated, it is only used by MIOpen (as of 28 August 2017).
 *
 * @param allotted_time
 * Time in seconds spent searching for a good Solution.
 * When this time is surpassed, no further Solutions are benchmarked,
 * and the best found is returned.
 *
 * @param command_queue
 * An OpenCL command queue
 *
 * @param a
 * Matrix A, memory will be unchanged
 *
 * @param b
 * Matrix B, memory will be unchanged
 * @param c
 * Matric C, memory will be unchanged
 *
 * @param enforce_determinism
 * If true, only kernels which are bitwise consistent are considered. Specifically, ICE=1.
 * For small m*n, enforce_determinism = false will find faster Solutions.
 *
 * @param tgg
 * The Geometry to find a Solution for
 *
 * @param verbose
 * Print the search summary to std::cout
 *
 * @param with_warnings
 * no longer relevant, to be removed in future versions.
 *
 * @return
 * The best Solution found during search
 */
Solution find(float            allotted_time,
              cl_command_queue command_queue,
              cl_mem           a,
              cl_mem           b,
              cl_mem           c,
              bool             enforce_determinism,
              const Geometry&  tgg,
              bool             verbose,
              bool             with_warnings);
}

#endif
