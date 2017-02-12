#include <vector>
#include <iostream>
#include <tinygemm/tinygemmerror.hpp>
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

#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/redirection.hpp>
#include <tinygemm/outputwriter.hpp>
#include <tinygemm/floattostring.hpp>
#include <tinygemm/consistencychecks.hpp>
#include <tinygemm/slowcpugemm.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/openclutil.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/accuracytests.hpp>
#include <tinygemm/devtinygemm.hpp>
#include <tinygemm/tinygemmgeometry.hpp>




namespace tinygemm{
namespace dev{




template <typename TFloat>
class Gemini{
  
public:
  tinygemm::TinyGemmGeometry gg;
  TinyGemmOffsets toff;
  const TFloat * a;
  const TFloat * b;
  const TFloat * c;
 

private:
  std::vector<TFloat> c_copy;
  std::vector<TFloat> c_for_cpu_compute;
  std::string outputfilename;
  outputwriting::OutputWriter mowri;

  openclutil::TinyGemmCommandQueueInContext tgcq;
  openclutil::SafeClMem a_gpu_safemem;
  openclutil::SafeClMem b_gpu_safemem;
  openclutil::SafeClMem c_gpu_safemem;
  openclutil::SafeClMem workspace_safemem;
  
  
  
public:
  
  Gemini(tinygemm::TinyGemmGeometry gg_, tinygemm::TinyGemmOffsets toff_, const TFloat * a_, const TFloat * b_, const TFloat * c_, bool verbose_, std::string outputfilename_):
  gg(gg_), toff(toff_), a(a_), b(b_), c(c_), outputfilename(outputfilename_), mowri(verbose_, outputfilename_.compare("") != 0, outputfilename_), tgcq(mowri, "tiny gemm command queue in devtinygemm"),  a_gpu_safemem("a_gpu_safemem, of Gemini"), b_gpu_safemem("b_gpu_safemem, of Gemini"), c_gpu_safemem("c_gpu_safemem, of Gemini"), workspace_safemem("workspace_safemem, of Gemini")
  
  {
    consistencychecks::check_ldx_mnk_consistent(gg);
    if (gg.derived.float_size_bytes != sizeof(TFloat)){
      throw tinygemm_error("float sizes don't agree in derivedtinygemm.cpp");
    }
    sizingup::check_sizes_ok_for_unsigned(gg, toff); 
    c_copy.resize(get_c_memsize()/sizeof(TFloat));
    std::memcpy(c_copy.data(), c, get_c_memsize());
    opencl_memory_initialise();
    
    
  }

  size_t get_c_memsize(){
    auto c_memsize = sizingup::get_n_elements_padded(gg.m, gg.n, gg.ldc, gg.isColMajor, gg.tC, toff.oc)*sizeof(TFloat);
    return c_memsize;
  }

  size_t get_a_memsize(){
    return sizingup::get_n_elements_padded(gg.m, gg.k, gg.lda, gg.isColMajor, gg.tA, toff.oa)*sizeof(TFloat);
  }
  
  size_t get_b_memsize(){
    return sizingup::get_n_elements_padded(gg.k, gg.n, gg.ldb, gg.isColMajor, gg.tB, toff.ob)*sizeof(TFloat);
  }
  
  size_t get_workspace_memsize(){
    return (gg.workspace_size + toff.oworkspace)*sizeof(TFloat);
  }
  
  
   
  
  void opencl_memory_initialise(){
    

    /* allocate memory for a,b,c on device, send it over */
    a_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_ONLY, get_a_memsize(), NULL, "a_gpu in devtinygemm");
    b_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_ONLY, get_b_memsize(), NULL, "b_gpu in devtinygemm");
    c_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_WRITE, get_c_memsize(), NULL, "c_gpu in devtinygemm");  
    workspace_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_WRITE, get_workspace_memsize(), NULL, "workspace_gpu in devtinygemm");     


          
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, a_gpu_safemem.clmem, CL_TRUE, 0, get_a_memsize(), a, 0, NULL, NULL, "enqueueing a on opencl_memory_initialise");
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, b_gpu_safemem.clmem, CL_TRUE, 0, get_b_memsize(), b, 0, NULL, NULL, "enqueueing b on opencl_memory_initialise");
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c, 0, NULL, NULL, "enqueueing c on opencl_memory_initialise");

  }

    
  void benchgemm(const std::vector<hyperparams::HyperParams> & hps, size_t number_of_runs){
    /* dev code's connection to tinygemm */
    
    

    tinygemm::benchgemm(tgcq.command_queue, hps, number_of_runs, gg, toff, a_gpu_safemem.clmem, b_gpu_safemem.clmem, c_gpu_safemem.clmem, workspace_safemem.clmem, true, mowri.filename, false);
  }
  
  tinygemm::TinyGemmSolution find(float allotted_time, bool enforce_deterministic){
    /* dev code's connection to tinygemm */
    tinygemm::TinyGemmSolution tgs = tinygemm::find(
      allotted_time, tgcq.command_queue, a_gpu_safemem.clmem, b_gpu_safemem.clmem, c_gpu_safemem.clmem, workspace_safemem.clmem, enforce_deterministic, gg, toff, 
      true, // yes, write to terminal (may be captured further upstream)
      outputfilename, // file where to write the output (if "", nowhere). A new mowri will be created in the function called. 
      false); 
   return tgs;
  }
  
  void accuracy_test(const hyperparams::HyperParams & hp, const TFloat * c_true_for_test){
    clEnqueueWriteBuffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c, 0, NULL, NULL);
    benchgemm({ hp }, 1);
    cl_event event_read_c_back;
    openclutil::cl_enqueue_read_buffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c_copy.data(), 0, NULL, &event_read_c_back, "enqueue read to c, in base_basegemm_with_accuracy_test");
    
    if (c_true_for_test == nullptr){
      c_for_cpu_compute.resize(get_c_memsize()/sizeof(TFloat));
      std::memcpy(c_for_cpu_compute.data(), c, get_c_memsize());
      slowcpugemm::gemms_cpu<TFloat>(gg, toff, a, b, c_for_cpu_compute.data(), tinygemm::default_alpha, tinygemm::default_beta, {"3fors"}, mowri);
      c_true_for_test = c_for_cpu_compute.data();
    }
    
    openclutil::cl_wait_for_events(1, &event_read_c_back, "waiting in accuracy test, dev tiny gemm");
    accuracytests::elementwise_compare(c, tinygemm::default_beta, c_true_for_test, c_copy.data(), c_copy.size(), mowri);
  }
};



template <typename TFloat>
void benchgemm(const std::vector<hyperparams::HyperParams> & hps,         
unsigned n_runs, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, const TFloat * a, const TFloat * b, const TFloat * c, bool verbose, std::string logfile){
  
  
  Gemini <TFloat> gem(gg, toff, a, b, c, verbose, logfile);
  gem.benchgemm(hps, n_runs);
}
template void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, const float * a, const float * b, const float * c, bool verbose, std::string logfile);

template void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, const double * a, const double * b, const double * c, bool verbose, std::string logfile);



template <typename TFloat>
void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, const TFloat * a, const TFloat * b,
const TFloat * c, const TFloat * c_true_for_test, bool verbose, std::string logfile){
  
  Gemini <TFloat> gem(gg, toff, a, b, c, verbose, logfile);
  gem.accuracy_test(hp, c_true_for_test);
}
template void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, const float * a, const float * b, const float * c, const float * c_true_for_test, bool verbose, std::string logfile);

template void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, const double * a, const double * b, const double * c, const double * c_true_for_test, bool verbose, std::string logfile);




template <typename TFloat>
tinygemm::TinyGemmSolution find(float allotted_time, const TFloat * a, const TFloat * b, const TFloat * c, bool enforce_deterministic, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff,  bool verbose, std::string logfile){
  
  Gemini <TFloat> gem(gg, toff, a, b, c, verbose, logfile);
  return gem.find(allotted_time, enforce_deterministic);
}
template tinygemm::TinyGemmSolution find(float allotted_time, const double * a, const double * b, const double * c, bool enforce_deterministic, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, bool verbose, std::string logfile);

template tinygemm::TinyGemmSolution find(float allotted_time, const float * a, const float * b, const float * c, bool enforce_deterministic, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, bool verbose, std::string logfile);






}
}




