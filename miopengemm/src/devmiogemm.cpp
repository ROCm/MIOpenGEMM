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
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/openclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/slowcpugemm.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/setabcw.hpp>

namespace MIOpenGEMM
{
namespace dev
{

template <typename TFloat>
class Gemini
{

  public:
  Geometry      gg;
  Offsets       toff;

  // a, b and c cpu memories. 
  std::vector<const TFloat *> cpu_mem;

  private:
  std::vector<TFloat>          c_copy;
  std::vector<TFloat>          c_for_cpu_compute;
  outputwriting::OutputWriter& mowri;


  openclutil::CommandQueueInContext tgcq;
  
  // a, b, c and workspace, gpu memories. 
  std::vector<openclutil::SafeClMem> gpu_safemem;

  // sizes of a, b, c and workspace gpu memories.
  std::vector<size_t> mem_size;
    
  // read write permissions of gpu data. 
  std::vector<cl_mem_flags> rw_perms;
  
  public:
  Gemini(Geometry                     gg_,
         Offsets                      toff_,
         const TFloat*                a_,
         const TFloat*                b_,
         const TFloat*                c_,
         outputwriting::OutputWriter& mowri_)
    : gg(gg_),
      
      toff(toff_),
      cpu_mem(Mat::E::N),
      mowri(mowri_),
      tgcq(mowri, "command queue of Gemini"),
      gpu_safemem(Mem::E::N, std::string("gpu_safemem vector in devmio")),
      mem_size(Mem::E::N),
      rw_perms(Mem::E::N)
  {
      cpu_mem[Mat::E::A] = a_;
      cpu_mem[Mat::E::B] = b_;
      cpu_mem[Mat::E::C] = c_;

      // TODO : these should be in enums in enum.hpp
      rw_perms[Mem::E::A] = CL_MEM_READ_ONLY;
      rw_perms[Mem::E::B] = CL_MEM_READ_ONLY;
      rw_perms[Mem::E::C] = CL_MEM_READ_WRITE;    
      rw_perms[Mem::E::W] = CL_MEM_READ_WRITE;    
  
      for (auto emem : {Mem::E::A, Mem::E::B, Mem::E::C}){
        mem_size[emem] = get_mat_memsize(emem);
      }
      mem_size[Mem::E::W] = get_workspace_memsize();
      
      gg.check_ldx_consistent();
      if (gg.derived.float_size_bytes != sizeof(TFloat))
      {
        throw miog_error("float sizes don't agree in Gemini");
      }
      c_copy.resize(mem_size[Mem::E::C] / sizeof(TFloat));
      std::memcpy(c_copy.data(), cpu_mem[Mem::E::C], mem_size[Mem::E::C]);
      opencl_memory_initialise();
    
    
  }
  
  
  size_t get_mat_memsize(Mem::E emem){
    Mat::E emat = Mat::mem_to_mat(emem);
    return sizeof(TFloat)*(gg.get_padded_area(emat) + toff.offsets[emem] + toff.tails[emem]);
  }

  size_t get_workspace_memsize() { return (gg.workspace_size + toff.offsets[Mem::E::W] + toff.tails[Mem::E::W]) * sizeof(TFloat); }

  void opencl_memory_initialise()
  {

    // allocate memory for a,b,c on device, send it over    
    for (auto emem : {Mem::E::A, Mem::E::B, Mem::E::C, Mem::E::W}){
      //Mat::E emat = Mat::mem_to_mat(emem);
      std::stringstream hash;
      
      hash << "GPU Mem " << Mem::M.name[emem] << " (Gemini) "
       << "with memory size " << mem_size[emem]  << ".";
      
      if (mem_size[emem] > 0){
        openclutil::cl_set_buffer_from_command_queue(gpu_safemem[emem].clmem,
                                                 tgcq.command_queue,
                                                 rw_perms[emem],
                                                 mem_size[emem],
                                                 NULL,
                                                 hash.str(),
                                                 true);
      }
    }
    
    
    for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C}){
      Mem::E emem = Mem::mat_to_mem(emat);
      openclutil::cl_enqueue_write_buffer(tgcq.command_queue,
                                        gpu_safemem[emem].clmem,
                                        CL_TRUE,
                                        0,
                                        mem_size[emem],
                                        cpu_mem[emat],
                                        0,
                                        NULL,
                                        NULL,
                                        std::string("enqueueing ") + Mat::M.name[emat] + " write buff ",
                                        true);
   }
                                        
  }

  void benchgemm(const std::vector<std::string>& hyperstrings, size_t max_number_of_runs, double max_time_per_kernel)
  {

    // dev code's connection to the outside
    std::vector<hyperparams::HyperParams> hps;
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

  Solution find(const FindParams& find_params, std::string constraints_string)
  {
    // dev code's connection to the outside
    bool     c_is_const        = false;
    Solution tgs               = MIOpenGEMM::find(tgcq.command_queue,
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

  void accuracy_test(const std::string& hyperstring, const TFloat* c_true_for_test)
  {
    clEnqueueWriteBuffer(
      tgcq.command_queue, gpu_safemem[Mem::E::C].clmem, CL_TRUE, 0,  mem_size[Mem::E::C], cpu_mem[Mat::E::C], 0, NULL, NULL);

    benchgemm({hyperstring}, 1, 1e12);

    cl_event event_read_c_back;
    openclutil::cl_enqueue_read_buffer(tgcq.command_queue,
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

    if (c_true_for_test == nullptr)
    {
      c_for_cpu_compute.resize(mem_size[Mem::E::C] / sizeof(TFloat));
      std::memcpy(c_for_cpu_compute.data(), cpu_mem[Mat::E::C], mem_size[Mem::E::C]);

      slowcpugemm::gemms_cpu<TFloat>(
        gg, toff, cpu_mem[Mat::E::A], cpu_mem[Mat::E::B], c_for_cpu_compute.data(), default_alpha, default_beta, {"3fors"}, mowri);
      c_true_for_test = c_for_cpu_compute.data();
    }

    openclutil::cl_wait_for_events(
      1, &event_read_c_back, "waiting in accuracy test, dev tiny gemm", true);
    accuracytests::elementwise_compare(
      cpu_mem[Mat::E::C], default_beta, c_true_for_test, c_copy.data(), c_copy.size(), mowri);
  }
};

template <typename TFloat>
void benchgemm(const std::vector<std::string>& hyperstrings,
               size_t                        max_n_runs,
               double max_time,
               const Geometry&                 gg,
               const Offsets&                  toff,
               const TFloat*                   a,
               const TFloat*                   b,
               const TFloat*                   c,
               outputwriting::OutputWriter&    mowri)
{
  Gemini<TFloat> gem(gg, toff, a, b, c, mowri);
  gem.benchgemm(hyperstrings, max_n_runs, max_time);
}

template void benchgemm(const std::vector<std::string>& hyperstrings,
                        size_t                        max_n_runs,
                        double max_time,
                        const Geometry&                 gg,
                        const Offsets&                  toff,
                        const float*                    a,
                        const float*                    b,
                        const float*                    c,
                        outputwriting::OutputWriter&    mowri);

template void benchgemm(const std::vector<std::string>& hyperstrings,
                        size_t                        max_n_runs,
                        double                        max_time,
                        const Geometry&                 gg,
                        const Offsets&                  toff,
                        const double*                   a,
                        const double*                   b,
                        const double*                   c,
                        outputwriting::OutputWriter&    mowri);

template <typename TFloat>
void accuracy_test(const std::string&           hyperstring,
                   const Geometry&              gg,
                   const Offsets&               toff,
                   const TFloat*                a,
                   const TFloat*                b,
                   const TFloat*                c,
                   const TFloat*                c_true_for_test,
                   outputwriting::OutputWriter& mowri)
{
  Gemini<TFloat> gem(gg, toff, a, b, c, mowri);
  gem.accuracy_test(hyperstring, c_true_for_test);
}

template void accuracy_test(const std::string&           hyperstring,
                            const Geometry&              gg,
                            const Offsets&               toff,
                            const float*                 a,
                            const float*                 b,
                            const float*                 c,
                            const float*                 c_true_for_test,
                            outputwriting::OutputWriter& mowri);

template void accuracy_test(const std::string&           hyperstring,
                            const Geometry&              gg,
                            const Offsets&               toff,
                            const double*                a,
                            const double*                b,
                            const double*                c,
                            const double*                c_true_for_test,
                            outputwriting::OutputWriter& mowri);

template <typename TFloat>
Solution find(

  const FindParams&            find_params,
  const TFloat*                a,
  const TFloat*                b,
  const TFloat*                c,
  std::string                  constraints_string,
  const Geometry&              gg,
  const Offsets&               toff,
  outputwriting::OutputWriter& mowri)
{

  Gemini<TFloat> gem(gg, toff, a, b, c, mowri);
  return gem.find(find_params, constraints_string);
}

template Solution find(const FindParams&            find_params,
                       const double*                a,
                       const double*                b,
                       const double*                c,
                       std::string                  constraints_string,
                       const Geometry&              gg,
                       const Offsets&               toff,
                       outputwriting::OutputWriter& mowri);

template Solution find(const FindParams&            find_params,
                       const float*                 a,
                       const float*                 b,
                       const float*                 c,
                       std::string                  constraints_string,
                       const Geometry&              gg,
                       const Offsets&               toff,
                       outputwriting::OutputWriter& mowri);
                       

template <typename TFloat>
Solution tbasicfind(const Geometry&   geometry,
                        const Offsets&    toff,
                        const FindParams& find_params,
                        outputwriting::OutputWriter& mowri,
                        std::string constraints_string){
    
    mowri << "generating cpu data ... " << Flush;
    std::vector<TFloat> v_a;
    std::vector<TFloat> v_b;
    std::vector<TFloat> v_c;
    std::vector<TFloat> v_workspace;
    setabcw::set_abcw(v_a, v_b, v_c, v_workspace, geometry, toff);
    mowri << "done." << Endl;
    return find(find_params, v_a.data(), v_b.data(), v_c.data(), constraints_string, geometry, toff, mowri);
}

Solution basicfind(
const FindParams& find_params,
                        std::string constraints_string,
                        const Geometry&   geometry,
                        const Offsets&    toff,
                        outputwriting::OutputWriter& mowri){



  if (geometry.floattype == 'f'){
    return tbasicfind<float>(geometry, toff, find_params, mowri, constraints_string);
  }
  
  else{
    return tbasicfind<double>(geometry, toff, find_params, mowri, constraints_string);
  }
}

                       

}
}
