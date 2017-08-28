/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <miopengemm/bundle.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/generic.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/programcache.hpp>
#include <miopengemm/timer.hpp>
namespace MIOpenGEMM
{

std::unordered_map<std::string, GemmKernelSquad> program_cache;

std::string get_device_name(const cl_device_id& device_id){

  //this does not slow performace (1000 -> 10000).
  std::string info_st(1000, ' ');
  size_t info_size(0);
  bool strict(true);
  
  oclutil::cl_set_device_info(device_id,
                            CL_DEVICE_NAME,
                            info_st.size(),
                            &info_st[0],
                            &info_size,
                            "obtaining CL_DEVICE_NAME in GemmCacheKey",
                            strict);
                            
  return info_st.substr(0, info_size - 1);

}

template <typename T>
GemmResult xgemm(bool              isColMajor,
               bool              tA,
               bool              tB,
               size_t            m,
               size_t            n,
               size_t            k,
               T             alpha,
               cl_mem            a,
               size_t            a_offset,
               size_t            lda,
               cl_mem            b,
               size_t            b_offset,
               size_t            ldb,
               T             beta,
               cl_mem            c,
               size_t            c_offset,
               size_t            ldc,
               cl_mem            w,
               size_t            w_offset,
               size_t            w_size,
               cl_command_queue* ptr_queue,
               cl_uint           num_events_in_wait_list,
               const cl_event*   event_wait_list,
               cl_event *        ptr_event_user)
{
  
  owrite::Writer silent_mowri(Ver::E::SILENT, "");
  
  cl_context   context;
  bool     tC{false};
  
  cl_device_id device_id;
  
  oclutil::cl_set_command_queue_info(*ptr_queue,
                                   CL_QUEUE_DEVICE,
                                   sizeof(cl_device_id),
                                   &device_id,
                                   nullptr,
                                   "GEMM",
                                   true);
  
  
  auto device_name = get_device_name(device_id);
  
  auto gemm_key = get_gemm_key(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, w_size, sizeof(T), device_name);

  if (program_cache.count(gemm_key) == 0)
  {

  oclutil::cl_set_command_queue_info(*ptr_queue,
                                        CL_QUEUE_CONTEXT,
                                        sizeof(cl_context),
                                        &context,
                                        nullptr,
                                        "GEMM",
                                        true);    

    size_t      rank = 0;
    Constraints constraints("");
    
    
    Geometry gg(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, w_size, get_floattype_char<T>());
    auto soln = get_default(*ptr_queue, gg, constraints, silent_mowri, IfNoCache::E::GENERIC, rank);
    program_cache[gemm_key] = GemmKernelSquad(soln.v_tgks, silent_mowri, context, device_id);
  }

  GemmKernelSquad& squad = program_cache[gemm_key]; 
  

  std::array<cl_mem, Mem::E::N> gpu_mems;
  std::array<size_t, Mem::E::N> offsets;

  gpu_mems[Mem::E::A] = a;
  gpu_mems[Mem::E::B] = b;
  gpu_mems[Mem::E::C] = c;
  gpu_mems[Mem::E::W] = w;

  offsets[Mem::E::A] = a_offset;
  offsets[Mem::E::B] = b_offset;
  offsets[Mem::E::C] = c_offset;
  offsets[Mem::E::W] = w_offset;

  squad.set_args(gpu_mems, offsets, &alpha, &beta, sizeof(T));
  std::vector<oclutil::SafeEvent> safe_events;

  for (size_t i = 0; i < squad.ptr_kernels.size() - 1; ++i){
    safe_events.emplace_back("Safe event GEMM");
    squad.ptr_kernels[i]->ptr_event = &safe_events[i].clevent;
  }
  squad.ptr_kernels.back()->ptr_event = ptr_event_user;

  auto oclr = run_kernels(*ptr_queue, squad.ptr_kernels, squad.v_wait_indices, num_events_in_wait_list, event_wait_list);


  return {};
}


template GemmResult xgemm<float>(bool, bool, bool, size_t, size_t, size_t, float, cl_mem, size_t, size_t, cl_mem, size_t, size_t, float, cl_mem, size_t, size_t, cl_mem, size_t, size_t, cl_command_queue*, cl_uint, const cl_event*, cl_event*);

template GemmResult xgemm<double>(bool, bool, bool, size_t, size_t, size_t, double, cl_mem, size_t, size_t, cl_mem, size_t, size_t, double, cl_mem, size_t, size_t, cl_mem, size_t, size_t, cl_command_queue*, cl_uint, const cl_event*, cl_event*);

}
