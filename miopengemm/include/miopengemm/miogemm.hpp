/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_MIOGEMM_HPP
#define GUARD_MIOPENGEMM_MIOGEMM_HPP

#include <miopengemm/findparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/platform.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{

/*! @brief
 * This function is being phased-out, it is only used by MIOpen (as of 28 August 2017).
 * Find a Solution, considering only Solutions which do not use workspace.
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
 *
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
 * Print the search summary to std :: cout
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

/*! @brief
 * Return 1 of 3 possible HyPas's,
 * this is a fallback function when better Solutions cannot be found
 */
HyPas get_generic(const Geometry& gg, const Constraints& constraints);

/*! @brief
 * Find and return a Solution which matches well the device and Geometry,
 * without performing any compiling-benchmarking.
 *
 * @param devinfo
 * contains device information, in particulat the device's name
 *
 * @param gg
 * GEMM Geometry
 *
 * @param mowri
 * Object which prints information about the Solution found and returned
 *
 * @param enoc
 * If there is no good cached match, a Solution will be returned depending on this parameter.
 * The options are to randomly select a viable Solution, or to use get_generic.
 */
// try and get a solution from cache, if all else fails get_generic.
Solution get_default_soln(const oclutil::DevInfo& devinfo,
                          const Geometry&         gg,
                          const Constraints&      constraints,
                          owrite::Writer&         mowri,
                          IfNoCache::E            enoc,
                          size_t                  rank);

/*! This function is being phased-out, it is only used by MIOpen (as of 28 August 2017)
 * [ the HIP branch of MIOpen currently calls this function ]
 *
 * @param gg                  GEMM problem Geometry
 * @return                    generic Solution
 */

Solution get_default(const Geometry& gg);
}

#endif
