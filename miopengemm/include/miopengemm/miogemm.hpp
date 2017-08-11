/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_MIOGEMM_HPP
#define GUARD_MIOPENGEMM_MIOGEMM_HPP

#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{

static const double default_alpha = 0.415693029182345929;
static const double default_beta  = 0.273539340934809345;

/*! @brief Find a Solution, considering only kernels without workspace requirements.
 *
 * @param allotted_time       An upper bound in seconds on the time spent searching. When this time
 * @param command_queue       An OpenCL command queue
 *                            has expired, the best Solution found thus far is returned.
 * @param a                   Matrix A, memory will be unchanged
 * @param b                   Matrix B, memory will be unchanged
 * @param c                   Matric C, memory will be unchanged
 * @param enforce_determinism If true, only kernels which are bitwise consistent are considered.
 *                            For small GEMM problems enforce_determinism = False finds faster
 *                            Solutions.
 * @param tgg                 GEMM problem Geometry as in geometry.hpp
 * @param verbose             Print the search summary to std::cout
 * @param with_warnings       Print warnings to std::cerr. Warnings are provided when the
 *                            allotted_time is insufficient to find good Solutions.
 * @return                    The best Solution found during search
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

/*! @brief Find a Solution, considering kernels with and without workspace requirements. This
 * function is experimental.
 *
 * @param command_queue       OpenCL command queue
 * @param find_params         Parameters define duration and structure of search
 * @param a                   Matrix A
 * @param b                   Matrix B
 * @param c                   Matric C
 * @param workspace           Workspace memory
 * @param constraints_string  String describing the constraints on the kernels considered. It is a
 *                            substring of a hyperstring, defining with hyper-parameters have fixed
 *                            values
 * @param gg                  GEMM problem Geometry as in geometry.hpp
 * @param toff                The offsets of a,b,c and workspace
 * @param mowri               Controls writing to terminal and file
 * @param c_is_const          Whether matrix C may be modified
 * @param use_mowri_tracker   Controls writing to terminal of search summary
 * @return                    best Solution found during search
 */

Solution find(cl_command_queue             command_queue,
              const FindParams&            find_params,
              cl_mem                       a,
              cl_mem                       b,
              cl_mem                       c,
              cl_mem                       workspace,
              const std::string            constraints_string,
              const Geometry&              gg,
              const Offsets&               toff,
              outputwriting::OutputWriter& mowri,
              bool                         c_is_const,
              bool                         use_mowri_tracker);

/*! @brief Return a default Solution from cache, without exploring. If no Solution exists, an error
 * will be thrown.
 *
 * @param command_queue       OpenCL command queue, used to extract device name
 * @param constraints_string  String describing the constraints on the kernels considered. It is a
 *                            substring of a hyperstring, defining with hyper-parameters have fixed
 *                            values
 * @param gg                  GEMM problem Geometry
 * @param k_comment           If a Solution from a custom cache is to be returned, this can be a
 *                            non-empty string
 * @param mowri               Controls writing to terminal and file
 * @return                    best Solution found during search
 */

Solution get_default(cl_command_queue             command_queue,
                     std::string                  constraints_string,
                     const Geometry&              gg,
                     std::string                  k_comment,
                     outputwriting::OutputWriter& mowri);

/*! @brief Return a default Solution, independent of device. This version of get_default does not
 * require a Solution to exist in cache.
 *
 * @param gg                  GEMM problem Geometry
 * @return                    generic Solution
 */

Solution get_default(const Geometry& gg);

/*! @brief check if a Solution exists in cache.
 *
 * @param command_queue       OpenCL command queue, used to extract device name
 * @param constraints_string  String describing the constraints on the kernels considered. It is a
 *                            substring of a hyperstring, defining with hyper-parameters have fixed
 *                            values
 * @param gg                  GEMM problem Geometry
 * @param k_comment           If a a custom cache is to be checked, this can be a non-empty string
 * @return                    A tuple, first argument is true if a Solution exists is cache. The
 *                            second argument is a string describing, if a Solution does not exist,
 *                            possible reasons.
 */
std::tuple<bool, std::string> check_for_default(cl_command_queue command_queue,
                                                std::string     constraints_string,
                                                const Geometry& gg,
                                                std::string     k_comment);

/*! @brief Benchmark the performance of MIOpenGEMM kernel(s).
 *
 * @param command_queue       OpenCL command queue
 * @param hyperstring         String describing the GEMM kernel(s) to be benchmarked
 * @param gg                  The Geometry of the GEMM problem
 * @param toff                Offsets of a, b, c, workspace
 */
void benchgemm(cl_command_queue             command_queue,
               const std::string&           hyperstring,
               unsigned                     n_runs,
               const Geometry&              gg,
               const Offsets&               toff,
               cl_mem                       a,
               cl_mem                       b,
               cl_mem                       c,
               cl_mem                       workspace,
               outputwriting::OutputWriter& mowri,
               bool                         c_is_const = false);

}
#endif
