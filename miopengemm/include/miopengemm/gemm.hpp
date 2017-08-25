/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_GEMMAPI_HPP
#define GUARD_MIOPENGEMM_GEMMAPI_HPP

#include <CL/cl.h>

namespace MIOpenGEMM
{

class GemmResult
{
};

// version without eventWaitList.

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
