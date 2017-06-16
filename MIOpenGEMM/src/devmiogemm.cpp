#include <vector>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <thread>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include  <CL/cl.h> 

#include <MIOpenGEMM/stringutilbase.hpp>
#include <MIOpenGEMM/redirection.hpp>
#include <MIOpenGEMM/outputwriter.hpp>
#include <MIOpenGEMM/floattostring.hpp>
#include <MIOpenGEMM/slowcpugemm.hpp>
#include <MIOpenGEMM/sizingup.hpp>
#include <MIOpenGEMM/openclutil.hpp>
#include <MIOpenGEMM/miogemm.hpp>
#include <MIOpenGEMM/accuracytests.hpp>
#include <MIOpenGEMM/devmiogemm.hpp>
#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/hyperparams.hpp>
#include <MIOpenGEMM/error.hpp>




namespace MIOpenGEMM{
namespace dev{




template <typename TFloat>
class Gemini{
  
public:
  Geometry gg;
  Offsets toff;
  const TFloat * a;
  const TFloat * b;
  const TFloat * c;
 

private:
  std::vector<TFloat> c_copy;
  std::vector<TFloat> c_for_cpu_compute;
  outputwriting::OutputWriter & mowri;

  openclutil::CommandQueueInContext tgcq;
  openclutil::SafeClMem a_gpu_safemem;
  openclutil::SafeClMem b_gpu_safemem;
  openclutil::SafeClMem c_gpu_safemem;
  openclutil::SafeClMem workspace_safemem;
  
  
  
public:
  
  Gemini(Geometry gg_, Offsets toff_, const TFloat * a_, const TFloat * b_, const TFloat * c_, outputwriting::OutputWriter & mowri_):
  gg(gg_), toff(toff_), a(a_), b(b_), c(c_), mowri(mowri_), tgcq(mowri, "command queue of Gemini"),  a_gpu_safemem("a_gpu_safemem, of Gemini"), b_gpu_safemem("b_gpu_safemem, of Gemini"), c_gpu_safemem("c_gpu_safemem, of Gemini"), workspace_safemem("workspace_safemem, of Gemini")
  
  {
    //consistencychecks::check_ldx_mnk_consistent(gg);
    gg.check_ldx_consistent();
    if (gg.derived.float_size_bytes != sizeof(TFloat)){
      throw miog_error("float sizes don't agree in Gemini");
    }
    sizingup::check_sizes_ok_for_unsigned(gg, toff); 
    c_copy.resize(get_c_memsize()/sizeof(TFloat));
    std::memcpy(c_copy.data(), c, get_c_memsize());
    opencl_memory_initialise();
    
    
  }

  size_t get_c_memsize(){
    auto c_memsize = sizingup::get_n_elements_padded(gg.m, gg.n, gg.ldX[nsHP::matC], gg.isColMajor, gg.tX[nsHP::matC], toff.oc, toff.tail_off_c)*sizeof(TFloat);
    return c_memsize;
  }

  size_t get_a_memsize(){
    return sizingup::get_n_elements_padded(gg.m, gg.k, gg.ldX[nsHP::matA], gg.isColMajor, gg.tX[nsHP::matA], toff.oa, toff.tail_off_a)*sizeof(TFloat);
  }
  
  size_t get_b_memsize(){
    return sizingup::get_n_elements_padded(gg.k, gg.n, gg.ldX[nsHP::matB], gg.isColMajor, gg.tX[nsHP::matB], toff.ob, toff.tail_off_b)*sizeof(TFloat);
  }
  
  size_t get_workspace_memsize(){
    return (gg.workspace_size + toff.oworkspace)*sizeof(TFloat);
  }
  
  
   
  
  void opencl_memory_initialise(){
    

    /* allocate memory for a,b,c on device, send it over */
    a_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_ONLY, get_a_memsize(), NULL, "a_gpu in Gemini");
    b_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_ONLY, get_b_memsize(), NULL, "b_gpu in Gemini");
    c_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_WRITE, get_c_memsize(), NULL, "c_gpu in Gemini");  
    
    std::stringstream ss_hash;
    if (get_workspace_memsize() > 0){
      ss_hash << "workspace_gpu in Gemini, with workspace_memsize : (" << get_workspace_memsize() << "(bytes) )";
      workspace_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_WRITE, get_workspace_memsize(), NULL, ss_hash.str());     
    }


          
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, a_gpu_safemem.clmem, CL_TRUE, 0, get_a_memsize(), a, 0, NULL, NULL, "enqueueing a on opencl_memory_initialise");
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, b_gpu_safemem.clmem, CL_TRUE, 0, get_b_memsize(), b, 0, NULL, NULL, "enqueueing b on opencl_memory_initialise");
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c, 0, NULL, NULL, "enqueueing c on opencl_memory_initialise");

  }


  void benchgemm(const std::vector<std::string> & hyperstrings, size_t number_of_runs){
    
    /* dev code's connection to the outside */
    std::vector<hyperparams::HyperParams> hps;
    for (auto & hyperstring : hyperstrings){
      MIOpenGEMM::benchgemm(tgcq.command_queue, hyperstring, number_of_runs, gg, toff, a_gpu_safemem.clmem, b_gpu_safemem.clmem, c_gpu_safemem.clmem, workspace_safemem.clmem, mowri);
    }
  }
  

  Solution find(const FindParams & find_params, std::string constraints_string){
    /* dev code's connection to the outside */
    
    bool c_is_const = false;
    bool use_mowri_tracker = false;
    Solution tgs = MIOpenGEMM::find(
      tgcq.command_queue, 
      find_params,
      a_gpu_safemem.clmem, b_gpu_safemem.clmem, c_gpu_safemem.clmem, workspace_safemem.clmem, constraints_string, gg, toff, 
      mowri, c_is_const, use_mowri_tracker); 
   return tgs;
  }
  
  void accuracy_test(const std::string & hyperstring, const TFloat * c_true_for_test){
    clEnqueueWriteBuffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c, 0, NULL, NULL);

    benchgemm({ hyperstring }, 1);

    
    cl_event event_read_c_back;
    openclutil::cl_enqueue_read_buffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c_copy.data(), 0, NULL, &event_read_c_back, "enqueue read to c, in base_basegemm_with_accuracy_test");
    
    if (c_true_for_test == nullptr){
      c_for_cpu_compute.resize(get_c_memsize()/sizeof(TFloat));
      std::memcpy(c_for_cpu_compute.data(), c, get_c_memsize());

      
      slowcpugemm::gemms_cpu<TFloat>(gg, toff, a, b, c_for_cpu_compute.data(), default_alpha, default_beta, {"3fors"}, mowri);
      c_true_for_test = c_for_cpu_compute.data();
    }
    
    openclutil::cl_wait_for_events(1, &event_read_c_back, "waiting in accuracy test, dev tiny gemm");
    accuracytests::elementwise_compare(c, default_beta, c_true_for_test, c_copy.data(), c_copy.size(), mowri);
  }
};



template <typename TFloat>
void benchgemm(const std::vector<std::string> & hyperstrings,         
unsigned n_runs, const Geometry & gg, const Offsets & toff, const TFloat * a, const TFloat * b, const TFloat * c, outputwriting::OutputWriter & mowri){
  Gemini <TFloat> gem(gg, toff, a, b, c, mowri);
  gem.benchgemm(hyperstrings, n_runs);
}


template void benchgemm(const std::vector<std::string> & hyperstrings,
unsigned n_runs, const Geometry & gg, const Offsets & toff, const float * a, const float * b, const float * c, outputwriting::OutputWriter & mowri);


template void benchgemm(const std::vector<std::string> & hyperstrings,
unsigned n_runs, const Geometry & gg, const Offsets & toff, const double * a, const double * b, const double * c, outputwriting::OutputWriter & mowri);



template <typename TFloat>
void accuracy_test(const std::string & hyperstring, 
const Geometry & gg, const Offsets & toff, const TFloat * a, const TFloat * b,
const TFloat * c, const TFloat * c_true_for_test, outputwriting::OutputWriter & mowri){
  
  Gemini <TFloat> gem(gg, toff, a, b, c, mowri);
  gem.accuracy_test(hyperstring, c_true_for_test);
}

template void accuracy_test(const std::string & hyperstring, 
const Geometry & gg, const Offsets & toff, const float * a, const float * b, const float * c, 
const float * c_true_for_test, outputwriting::OutputWriter & mowri);


template void accuracy_test(const std::string & hyperstring, 
const Geometry & gg, const Offsets & toff, const double * a, const double * b, const double * c, const double * c_true_for_test, outputwriting::OutputWriter & mowri);




template <typename TFloat>
Solution find(

const FindParams & find_params, const TFloat * a, const TFloat * b, const TFloat * c, std::string constraints_string, const Geometry & gg, const Offsets & toff,  outputwriting::OutputWriter & mowri){
  
  Gemini <TFloat> gem(gg, toff, a, b, c, mowri);
  return gem.find(find_params, constraints_string);
}

template Solution find(const FindParams & find_params, const double * a, const double * b, const double * c, std::string constraints_string, const Geometry & gg, const Offsets & toff, outputwriting::OutputWriter & mowri);

template Solution find(const FindParams & find_params, const float * a, const float * b, const float * c,    std::string constraints_string, const Geometry & gg, const Offsets & toff, outputwriting::OutputWriter & mowri);




}
}
