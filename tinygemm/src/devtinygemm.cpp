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
#include <tinygemm/kernelchecks.hpp>
#include <tinygemm/kernelsnips.hpp>
#include <tinygemm/floattostring.hpp>
#include <tinygemm/consistencychecks.hpp>
#include <tinygemm/slowcpugemm.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/openclutil.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/betackernelutil.hpp>
#include <tinygemm/accuracytests.hpp>
#include <tinygemm/devtinygemm.hpp>
#include <tinygemm/tinygemmgeometry.hpp>




namespace tinygemm{
namespace dev{




template <typename TFloat>
class Gemini{
  
public:
  tinygemm::TinyGemmGeometry gg;
  const TFloat * a;
  const TFloat * b;
  const TFloat * c;
  TFloat alpha;
  TFloat beta;
 

private:
  std::vector<TFloat> c_copy;
  std::vector<TFloat> c_for_cpu_compute;
  std::string outputfilename;
  outputwriting::OutputWriter mowri;

  openclutil::TinyGemmCommandQueueInContext tgcq;
  openclutil::SafeClMem a_gpu_safemem;
  openclutil::SafeClMem b_gpu_safemem;
  openclutil::SafeClMem c_gpu_safemem;
  
public:
  
  Gemini(tinygemm::TinyGemmGeometry gg, const TFloat * a, const TFloat * b, const TFloat * c, bool verbose, TFloat alpha, TFloat beta, std::string outputfilename):
  gg(gg), a(a), b(b), c(c), alpha(alpha), beta(beta), outputfilename(outputfilename), mowri(verbose, outputfilename.compare("") != 0, outputfilename), tgcq(mowri, "tiny gemm command queue in devtinygemm"),  a_gpu_safemem("a_gpu_safemem, of Gemini"), b_gpu_safemem("b_gpu_safemem, of Gemini"), c_gpu_safemem("c_gpu_safemem, of Gemini")
  
  {
    consistencychecks::check_ldx_mnk_consistent(gg);
    sizingup::check_sizes_ok_for_unsigned(gg); 
    c_copy.resize(get_c_memsize()/sizeof(TFloat));
    std::memcpy(c_copy.data(), c, get_c_memsize());
    opencl_memory_initialise();
  }

  //void benchgemm_cpu(std::vector<std::string> cpu_algs, TFloat * c_on_cpu){
    //if (false){
      //slowcpugemm::gemms_cpu<TFloat>(gg, a, b, c_on_cpu, alpha, beta, cpu_algs, mowri);  
    //}
  //}


  size_t get_c_memsize(){
    auto c_memsize = sizingup::get_n_elements_padded(gg.m, gg.n, gg.ldc, gg.isColMajor, gg.tC, gg.c_offset)*sizeof(TFloat);
    return c_memsize;
  }

  size_t get_a_memsize(){
    return sizingup::get_n_elements_padded(gg.m, gg.k, gg.lda, gg.isColMajor, gg.tA, gg.a_offset)*sizeof(TFloat);
  }
  
  size_t get_b_memsize(){
    return sizingup::get_n_elements_padded(gg.k, gg.n, gg.ldb, gg.isColMajor, gg.tB, gg.b_offset)*sizeof(TFloat);
  }
  
  
   
  
  void opencl_memory_initialise(){
    
    /* allocate memory for a,b,c on device, send it over */
    a_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_ONLY, get_a_memsize(), NULL, "a_gpu in devtinygemm");
    b_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_ONLY, get_b_memsize(), NULL, "b_gpu in devtinygemm");
    c_gpu_safemem.clmem = openclutil::cl_create_buffer_from_command_queue(tgcq.command_queue, CL_MEM_READ_WRITE, get_c_memsize(), NULL, "c_gpu in devtinygemm");     
          
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, a_gpu_safemem.clmem, CL_TRUE, 0, get_a_memsize(), a, 0, NULL, NULL, "enqueueing a on opencl_memory_initialise");
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, b_gpu_safemem.clmem, CL_TRUE, 0, get_b_memsize(), b, 0, NULL, NULL, "enqueueing b on opencl_memory_initialise");
    openclutil::cl_enqueue_write_buffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c, 0, NULL, NULL, "enqueueing c on opencl_memory_initialise");    
    
  }

  /* dev code's placenta to the outside world */  
  void benchgemm(const std::vector<hyperparams::HyperParams> & hps, size_t number_of_runs){
    tinygemm::benchgemm(tgcq.command_queue, hps, number_of_runs, floattostring::get_float_char<TFloat>(), gg, alpha, beta, a_gpu_safemem.clmem, b_gpu_safemem.clmem, c_gpu_safemem.clmem, true, mowri.filename, false);
  }
  
  tinygemm::TinyGemmSolution find(float allotted_time, bool enforce_deterministic){
    tinygemm::TinyGemmSolution tgs = tinygemm::find(
      allotted_time, tgcq.command_queue, a_gpu_safemem.clmem, b_gpu_safemem.clmem, c_gpu_safemem.clmem, enforce_deterministic, floattostring::get_float_char<TFloat>(), gg, alpha, beta, 
      true, // yes, write to terminal (may be captured further upstream)
      outputfilename, // file where to write the output (if "", nowhere). A new mowri will be created in the function called. 
      false); 
   return tgs;
  }
  





  void accuracy_test(const hyperparams::HyperParams & hp, const TFloat * c_true_for_test){
    mowri << "in accuracy_test of devtinygemm" << Endl;
    mowri << "writing the initial c to c-gpu" << Endl;
    clEnqueueWriteBuffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c, 0, NULL, NULL);
    mowri << "running benchgemm, with c_is_const = false " << Endl;
    benchgemm({ hp }, 1);
    mowri << "reading c-gpu to c_copy" << Endl;
    cl_event event_read_c_back;
    openclutil::cl_enqueue_read_buffer(tgcq.command_queue, c_gpu_safemem.clmem, CL_TRUE, 0, get_c_memsize(), c_copy.data(), 0, NULL, &event_read_c_back, "enqueue read to c, in base_basegemm_with_accuracy_test");
    
    if (c_true_for_test == nullptr){
      mowri << "obtaining c_true_for_test (as passed in was a nullptr) with a cpu gemm" << Endl;
      c_for_cpu_compute.resize(get_c_memsize()/sizeof(TFloat));
      std::memcpy(c_for_cpu_compute.data(), c, get_c_memsize());
      slowcpugemm::gemms_cpu<TFloat>(gg, a, b, c_for_cpu_compute.data(), alpha, beta, {"3fors"}, mowri);
      c_true_for_test = c_for_cpu_compute.data();
    }
    
    mowri << "waiting for read back to complete" << Endl;
    openclutil::cl_wait_for_events(1, &event_read_c_back, "waiting un accuracy test, dev tiny gemm");


      
    
    mowri << "running accuract test : c_copy vs c_true_for_test" << Endl;
    accuracytests::elementwise_compare(c, beta, c_true_for_test, c_copy.data(), c_copy.size(), mowri);
    
    
    
    //accuracytests::accuracy_test(gg.isColMajor, gg.tC, gg.m, gg.n, gg.ldc, c_true_for_test, c_copy.data(), gg.c_offset, mowri, 1e-6);
  }
};



template <typename TFloat>
void benchgemm(const std::vector<hyperparams::HyperParams> & hps,         
unsigned n_runs, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const TFloat * a, const TFloat * b, const TFloat * c, bool verbose, std::string logfile){
  
  Gemini <TFloat> gem(gg, a, b, c, verbose, alpha, beta, logfile);
  gem.benchgemm(hps, n_runs);

}

template void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs, const tinygemm::TinyGemmGeometry & gg,const double alpha, const double beta, const float * a, const float * b, const float * c, bool verbose, std::string logfile);

template void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs, const tinygemm::TinyGemmGeometry & gg,const double alpha, const double beta, const double * a, const double * b, const double * c, bool verbose, std::string logfile);

template <typename TFloat>
void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const TFloat * a, const TFloat * b,
const TFloat * c, const TFloat * c_true_for_test, bool verbose, std::string logfile){
  
  Gemini <TFloat> gem(gg, a, b, c, verbose, alpha, beta, logfile);
  gem.accuracy_test(hp, c_true_for_test);

}

template void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const float * a, const float * b, const float * c, const float * c_true_for_test, bool verbose, std::string logfile);

template void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const double * a, const double * b, const double * c, const double * c_true_for_test, bool verbose, std::string logfile);


      
  //void benchgemm_gpu(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs, unsigned do_test,  const TFloat * c_true_for_test){

    //if (do_test != 0){
      //mowri << "about to perform test in Gemini benchgemm_gpu" << Endl;
      //for (auto & hp : hps){
        //base_basegemm_with_accuracy_test(hp);
      //}
    //}
    
    //else{
      //base_benchgemm(hps, n_runs);
    //}
    
    //if (do_test != 0){
      ///* compare */
      //accuracytests::accuracy_test(gg.isColMajor, gg.tC, gg.m, gg.n, gg.ldc, c_true_for_test, c, gg.c_offset, mowri, 1e-6);
    //}

  //}


//void hello(){
  //std::cout << "hello!" <<std::endl;
//}





//template <typename TFloat>
//void benchgemm(

//bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, TFloat alpha, const TFloat * a, unsigned lda, unsigned a_offset, const TFloat * b, unsigned ldb, unsigned b_offset, TFloat beta, TFloat * c, unsigned ldc, unsigned c_offset, 

//std::vector<std::string> cpu_algs, const std::vector<hyperparams::HyperParams> hps, 

//bool capture_output, std::string & output,
//const TFloat * c_true_for_test, unsigned do_test, 

//unsigned n_runs, 

//std::string outputfilename, 
//bool findfirst, 
//float allotted_time, 
//bool enforce_deterministic){
    
  //if (findfirst == true && hps.size() != 0){
    //throw tinygemm_error( "findfirst is true, and so gpu_kernel_strings should be an empty list \n");
  //}
  
  //if (findfirst == true && cpu_algs.size() != 0){
    //throw tinygemm_error("findfirst is true, and so cpu_algs should be an empty list \n ");
  //}
  
  //if (allotted_time >= 0 and findfirst == false){
    //throw tinygemm_error("allotted_time is positive, so findfirst was expected to be true. We are being pedantic here, but please set allotted_time to a negative value. Just checking that you're aware that allotted_time is specific to the find algorith");
  //}
  
  //tinygemm::TinyGemmGeometry gg(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset);  

  //std::stringstream buffer;
  //auto cout_buff = std::cout.rdbuf();
  //if (capture_output == true){
    //std::cout.rdbuf(buffer.rdbuf());
  //}
  //std::ofstream nowhere;
    
  //Gemini <TFloat> gem (gg, a, b, c, true, alpha, beta, outputfilename);
  //if (findfirst == true){

    //tinygemm::TinyGemmSolution tgs = gem.find(allotted_time, enforce_deterministic); 
    //gem.benchgemm_cpu(cpu_algs);
    //gem.benchgemm_gpu({tgs.hp}, n_runs, do_test, c_true_for_test);
  //}
  
  //else{
    //gem.benchgemm_cpu(cpu_algs);
    //gem.benchgemm_gpu(hps, n_runs, do_test, c_true_for_test);
  //}
  
  //if (capture_output == true){
    //output = buffer.str();
    //std::cout.rdbuf(cout_buff);
  //}
  
  //else{
    //output = "capture_output was false, so nothing here";
  //}
//}

//template void benchgemm(bool isColMajor, bool tA, bool tB, bool tC,  unsigned m, unsigned n, unsigned k, float alpha, const float * a, unsigned lda, unsigned a_offset, const float * b, unsigned ldb, unsigned b_offset, float beta, float * c, unsigned ldc, unsigned c_offset, std::vector<std::string> cpu_algs, std::vector<hyperparams::HyperParams > gpu_kernel_strings, bool capture_output, std::string & output, const float * c_true_for_test, unsigned do_test, unsigned n_runs, std::string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic);

//template void benchgemm(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, double alpha, const double * a, unsigned lda, unsigned a_offset, const double * b, unsigned ldb, unsigned b_offset, double beta, double * c, unsigned ldc, unsigned c_offset, std::vector<std::string> cpu_algs, std::vector<hyperparams::HyperParams > gpu_kernel_strings, bool capture_output, std::string & output, const double * c_true_for_test, unsigned do_test, unsigned n_runs, std::string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic);




}
}




