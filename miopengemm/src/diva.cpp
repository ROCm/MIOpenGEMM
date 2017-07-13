/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <miopengemm/accuracytests.hpp>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/floattostring.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/graph.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/slowcpugemm.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{
namespace dev
{

template <typename TFl>
void Diva<TFl>::initialise_cpu_mem(const TFl* a_, const TFl* b_, const TFl* c_)
{
  cpu_mem[Mat::E::A] = a_;
  cpu_mem[Mat::E::B] = b_;
  cpu_mem[Mat::E::C] = c_;
}

template <typename TFl>
void Diva<TFl>::initialise_common()
{


  // TODO : these could be enums in enum.hpp
  rw_perms[Mem::E::A] = CL_MEM_READ_ONLY;
  rw_perms[Mem::E::B] = CL_MEM_READ_ONLY;
  rw_perms[Mem::E::C] = CL_MEM_READ_WRITE;
  rw_perms[Mem::E::W] = CL_MEM_READ_WRITE;

  for (auto emem : {Mem::E::A, Mem::E::B, Mem::E::C})
  {
    mem_size[emem] = get_mat_memsize(gg, toff, emem);
  }
  mem_size[Mem::E::W] = get_workspace_memsize();

  gg.check_ldx_consistent();

  if (gg.derived.float_size_bytes != sizeof(TFl))
  {
    std::stringstream errm;
    errm << "float sizes don't agree in Diva. ";
    errm << "the size from geometry is " << gg.derived.float_size_bytes << ". ";
    errm << "the size from the template parameter is " << sizeof(TFl) << ".";
    throw miog_error(errm.str());
  }

  c_copy.resize(mem_size[Mem::E::C] / sizeof(TFl));
  std::memcpy(c_copy.data(), cpu_mem[Mem::E::C], mem_size[Mem::E::C]);
  opencl_memory_initialise();

}

template <typename TFl>
Diva<TFl>::Diva(Geometry gg_, Offsets toff_, owrite::Writer& mowri_, long)
  : gg(gg_),
    toff(toff_),
    cpu_mem(Mat::E::N),
    mowri(mowri_),
    tgcq(mowri, "command queue of Diva"),
    gpu_safemem(Mem::E::N, std::string("gpu_safemem vector of Diva")),
    mem_size(Mem::E::N),
    rw_perms(Mem::E::N)
{

}

template <typename TFl>
Diva<TFl>::Diva(Geometry                     gg_,
                Offsets                      toff_,
                const TFl*                   a_,
                const TFl*                   b_,
                const TFl*                   c_,
                owrite::Writer& mowri_)
  : Diva(gg_, toff_, mowri_, 42)

{

  initialise_cpu_mem(a_, b_, c_);
  initialise_common();

}

template <typename TFl>
void Diva<TFl>::initialise_cpu_mem_from_scratch()
{

  
  setabcw::set_abc(__cpu_mem[Mat::E::A], __cpu_mem[Mat::E::B], __cpu_mem[Mat::E::C], gg, toff);
    
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C}){
    cpu_mem[emat] = __cpu_mem[emat].data();
  }
  
}

template <typename TFl>
Diva<TFl>::Diva(Geometry gg_, Offsets toff_, owrite::Writer& mowri_)
  : Diva(gg_, toff_, mowri_, 42)

{

  initialise_cpu_mem_from_scratch();
  initialise_common();
}


template <typename TFl>
size_t Diva<TFl>::get_workspace_memsize()
{
  return (gg.workspace_size + toff.offsets[Mem::E::W] + toff.tails[Mem::E::W]) * sizeof(TFl);
}

template <typename TFl>
void Diva<TFl>::opencl_memory_initialise()
{

  // allocate memory for a,b,c on device, send it over
  for (auto emem : {Mem::E::A, Mem::E::B, Mem::E::C, Mem::E::W})
  {
    std::stringstream hash;

    hash << "GPU Mem " << Mem::M.name[emem] << " (Diva) "
         << "with memory size " << mem_size[emem] << ".";

    if (mem_size[emem] > 0)
    {
      oclutil::cl_set_buffer_from_command_queue(gpu_safemem[emem].clmem,
                                                   tgcq.command_queue,
                                                   rw_perms[emem],
                                                   mem_size[emem],
                                                   NULL,
                                                   hash.str(),
                                                   true);
    }
  }

  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    Mem::E emem = Mem::mat_to_mem(emat);
    oclutil::cl_enqueue_write_buffer(tgcq.command_queue,
                                        gpu_safemem[emem].clmem,
                                        CL_TRUE,
                                        0,
                                        mem_size[emem],
                                        cpu_mem[emat],
                                        0,
                                        NULL,
                                        NULL,
                                        std::string("enqueueing ") + Mat::M.name[emat] +
                                          " writebuff ",
                                        true);
  }
}

template <typename TFl>
void Diva<TFl>::benchgemm(const std::vector<std::string>& hyperstrings,
                          size_t                          max_number_of_runs,
                          double                          max_time_per_kernel)
{

  // dev code's connection to the outside
  std::vector<HyPas> hps;
  for (auto& hyperstring : hyperstrings)
  {
    MIOpenGEMM::benchgemm(tgcq.command_queue,
                          hyperstring,
                          max_number_of_runs,
                          max_time_per_kernel,
                          gg,
                          toff,
                          gpu_safemem[Mem::E::A].clmem,
                          gpu_safemem[Mem::E::B].clmem,
                          gpu_safemem[Mem::E::C].clmem,
                          gpu_safemem[Mem::E::W].clmem,
                          mowri);
  }
}

template <typename TFl>
Solution Diva<TFl>::find(const FindParams& find_params, std::string constraints_string)
{
  // dev code's connection to the outside
  bool     c_is_const = false;
 
 
  //zzzzzzzz 
  Solution tgs        = MIOpenGEMM::find(tgcq.command_queue,
                                  find_params,
                                  gpu_safemem[Mem::E::A].clmem,
                                  gpu_safemem[Mem::E::B].clmem,
                                  gpu_safemem[Mem::E::C].clmem,
                                  gpu_safemem[Mem::E::W].clmem,
                                  constraints_string,
                                  gg,
                                  toff,
                                  mowri,
                                  c_is_const);
  return tgs;
}

template <typename TFl>
void Diva<TFl>::accuracy_test(const std::string& hyperstring, const TFl* c_true_for_test)
{



  // copy the const cpu matrix to the gpu
  clEnqueueWriteBuffer(tgcq.command_queue,
                       gpu_safemem[Mem::E::C].clmem,
                       CL_TRUE,
                       0,
                       mem_size[Mem::E::C],
                       cpu_mem[Mat::E::C],
                       0,
                       NULL,
                       NULL);

  
  // run gemm once on the gpu
  benchgemm({hyperstring}, 1, 1e12);


  // read the result to c_copy on the cpu
  cl_event event_read_c_back;
  oclutil::cl_enqueue_read_buffer(tgcq.command_queue,
                                     gpu_safemem[Mat::E::C].clmem,
                                     CL_TRUE,
                                     0,
                                     mem_size[Mem::E::C],
                                     c_copy.data(),
                                     0,
                                     NULL,
                                     &event_read_c_back,
                                     "enqueue read to c, in base_basegemm_with_accuracy_test",
                                     true);

  // if the user has not provided the correct answer, compute it 
  if (c_true_for_test == nullptr)
  {
    // run gemm on the cpu, result stored in c_for_cpu_compute.
    c_for_cpu_compute.resize(mem_size[Mem::E::C] / sizeof(TFl));
    std::memcpy(c_for_cpu_compute.data(), cpu_mem[Mat::E::C], mem_size[Mem::E::C]);

    slowcpugemm::gemms_cpu<TFl>(gg,
                                toff,
                                cpu_mem[Mat::E::A],
                                cpu_mem[Mat::E::B],
                                c_for_cpu_compute.data(),
                                default_alpha,
                                default_beta,
                                {"3fors"},
                                mowri);
                                
    c_true_for_test = c_for_cpu_compute.data();
  }

  
  // make sure the gpu gemm is complete
  oclutil::cl_wait_for_events(
    1, &event_read_c_back, "waiting in accuracy test for gpu gemm to complete ", true);
  

  // compare cpu and gpu results 
  accuracytests::elementwise_compare(
    cpu_mem[Mat::E::C], default_beta, c_true_for_test, c_copy.data(), c_copy.size(), mowri);
}

template class Diva<float>;
template class Diva<double>;
}
}
