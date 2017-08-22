/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <miopengemm/gemm.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM{


// TODO : make thread safe version
// TODO : include command queue in map key.

class KernelSquad{
  public:
  std::vector<Kernel> kernels;
  std::vector<Kernel*> ptr_kernels;
  std::vector<std::vector<size_t>> v_wait_indices;
};


std::unordered_map<std::string, KernelSquad> gemm_cache ;



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
cl_event* event){
             
  size_t wSpaceSize {0};
  bool tC {false};
  Geometry gg(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, wSpaceSize, 'f');
  auto gg_string = gg.get_string();
  
  if (gemm_cache.count(gg_string) == 0){
    // build, cache.
  }
  
  run_kernels(gemm_cache[gg_string].ptr_kernels, gemm_cache[gg_string].v_wait_indices);
}


}
