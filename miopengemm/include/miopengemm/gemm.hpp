/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_GEMMAPI_HPP
#define GUARD_MIOPENGEMM_GEMMAPI_HPP

#include <CL/cl.h>

namespace MIOpenGEMM
{

// TODO : how to get classes showing in documentation.


/*! @brief 
 *  The return type from xgemm  */
class GemmStatus
{
  public:
  
  /*! true if GEMM ran successfully, otherwise false */
  bool success;
  
  /*! A non-negative integer, identifying where the GemmKernelSquad used is privately cached. 
   * It should be used as an argument to xgemm in subsequent calls to GEMM 
   * with the same (device, geometry) as long as there are no other threads simultaneously 
   * running GEMM on the same (device, geometry) */
  int  ID;
  
  GemmStatus(bool x, int ID_) : success(x), ID(ID_) {}
};


/*! @brief
 * GEneral Matric Multiplication.
 * - \f$ C \leftarrow \alpha op(A) op(B) + \beta C \f$
 * To get started with understanding GEMM parameters
 * isColMajor, tA, tB, m, n, k lda, ldb, ldc, alpha and beta see (TODO).
 * 
 * @param a
 * memory buffer for matrix A
 * 
 * @param b
 * memory buffer for matrix B
 * 
 * @param c
 * memory buffer for matrix C
 * 
 * @param w
 * workspace memory buffer. Should be \b nullptr if GEMM is inplace
 * 
 * @param a_offset
 * The number of elements of type T before the first matrix element in cl_mem buffers a   
 * 
 * @param w_size
 * The number of elements of type T which are usable in w. 
 * Usable elements are in range w_offset ... w_offset + w_size - 1
 * 
 * @param ptr_queue
 * pointer to the cl_command_queue on which to enqueue 
 * 
 * @param event_wait_list
 * The cl_events which must complete before GEMM begins 
 * 
 * @param ptr_event
 * The event associated with the (final) GEMM kernel. When this event completes, GEMM is complete
 * 
 * @param ID
 * Define a GEMM geometry to be any (isColMajor, tA, tB, m, n, k lda, ldb, ldc, w_size) tuple.
 * The first time GEMM is run for a particular (device, geometry) pair, ID must be negative. 
 * Thereafter, the ID of the GemmStatus returned can be used for this (device, geometry), 
 * as long as ther same (device, geometry) is not being run simultaneously on a different thread. 
 * If it is being run simultaneously on a different thread, ID < 0 should be passed, 
 * to obtain a new ID from the returned GemmStatus. See (TODO) for clarification
 *  
 * @return
 * A GemmStatus.
 */
 
template <typename T>
GemmStatus xgemm(bool              isColMajor,
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
                 cl_event*         ptr_event,
                 int               ID);


/*! @brief
 * Free memory of GEMM ID. Calling this function is not required, 
 * but it can be used to reclaim memory early if needed. 
 * After free(ID) is called, ID is no longer valid for xgemm.   
 */
void free(size_t ID);

}

#endif
