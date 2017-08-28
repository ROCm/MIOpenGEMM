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
              

// document this function
class GemmResult
{
};

template <typename T>
GemmResult xgemm(
               bool              isColMajor,
               bool              tA,
               bool              tB,
               size_t            m,
               size_t            n,
               size_t            k,
               T                 alpha,
               cl_mem            a,
               size_t            a_offset,
               size_t            lda,
               cl_mem            b,
               size_t            b_offset,
               size_t            ldb,
               T                 beta,
               cl_mem            c,
               size_t            c_offset,
               size_t            ldc,
               cl_mem            w,
               size_t            w_offset,
               size_t            w_size,
               cl_command_queue* ptr_queue,
               cl_uint           num_events_in_wait_list,
               const cl_event*   event_wait_list,
               cl_event*         ptr_event);
}              


#endif
