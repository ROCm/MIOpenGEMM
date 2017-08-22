/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_GEMMAPI_HPP
#define GUARD_MIOPENGEMM_GEMMAPI_HPP

#include <CL/cl.h>

namespace MIOpenGEMM
{
  
class GemmResult{
  
};

//version without eventWaitList.
GemmResult f32(
bool isColMajor, 
bool tA, 
bool tB, 
size_t m, 
size_t n, 
size_t k, 
float alpha,
cl_mem a, size_t a_offset, size_t lda,
cl_mem b, size_t b_offset, size_t ldb,
float beta,
cl_mem c, size_t c_offset, size_t ldc,
cl_command_queue* queue, 
cl_event* event = nullptr);

}




#endif
