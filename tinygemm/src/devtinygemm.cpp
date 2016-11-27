#include <vector>
#include <iostream>
#include "tinygemmerror.hpp"
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
#include "tinygemmerror.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include  <CL/cl.h> 

#include "stringutilbase.hpp"
#include "redirection.hpp"
#include "outputwriter.hpp"
#include "kernelchecks.hpp"
#include "kernelsnips.hpp"
#include "floattostring.hpp"
#include "consistencychecks.hpp"
#include "slowcpugemm.hpp"
#include "sizingup.hpp"
#include "openclutil.hpp"
#include "tinygemm.hpp"
#include "betackernelutil.hpp"
#include "accuracytests.hpp"
#include "devtinygemm.hpp"
#include "tinygemmgeometry.hpp"
#include "defaultoutpath.hpp"




namespace tinygemm{
namespace dev{






template <typename TFloat>
class Gemini{
  
public:
  tinygemm::TinyGemmGeometry gg;
  const TFloat * a;
  const TFloat * b;
  TFloat * c;
  TFloat alpha;
  TFloat beta;
 

private:
  std::vector<TFloat> c_start_copy;
  std::string outputfilename;
  outputwriting::OutputWriter mowri;
  
  
  /* opencl boilerplate */
  cl_int ret;
  cl_platform_id platform = nullptr;
  cl_uint num_platforms;
  cl_context context;
  cl_device_id device_id_to_use;
  cl_command_queue command_queue;
  cl_mem a_gpu = NULL;
  cl_mem b_gpu = NULL;
  cl_mem c_gpu = NULL;
  
  
  
public:
  
  Gemini(tinygemm::TinyGemmGeometry gg, const TFloat * a, const TFloat * b, TFloat * c, TFloat alpha, TFloat beta, std::string outputfilename):gg(gg), a(a), b(b), c(c), alpha(alpha), beta(beta), outputfilename(outputfilename), mowri(true, outputfilename.compare("") != 0, outputfilename)  {
  
    
    consistencychecks::check_ldx_mnk_consistent(gg);
    sizingup::check_sizes_ok_for_unsigned(gg); 
    c_start_copy.resize(get_c_memsize()/sizeof(TFloat));
    std::memcpy(c_start_copy.data(), c, get_c_memsize());
    
    
    opencl_memory_initialise();
  
  }

  void benchgemm_cpu(std::vector<std::string> cpu_algs){
    if (false){
      slowcpugemm::gemm_3fors_cpu<TFloat>(gg.isColMajor, gg.tA, gg.tB, gg.tC, gg.lda, gg.ldb, gg.ldc, gg.m, gg.n, gg.k, a + gg.a_offset, b + gg.b_offset, c + gg.c_offset, alpha, beta, cpu_algs, mowri);  
    }
  }


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
    
    openclutil::set_platform_etc(platform, num_platforms, context, device_id_to_use, mowri);
    
    /* Create an inorder command queue with profiling enabled. Profiling is enabled so that we can get start and end times for kernels*/
    command_queue = clCreateCommandQueue(context, device_id_to_use, CL_QUEUE_PROFILING_ENABLE, &ret);
    
    /* allocate memory for a,b,c on device, send it over */
    a_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, get_a_memsize(), NULL, &ret);
    b_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, get_b_memsize(), NULL, &ret);
    c_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, get_c_memsize(), NULL, &ret);      
    clEnqueueWriteBuffer(command_queue, a_gpu, CL_TRUE, 0, get_a_memsize(), a, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, b_gpu, CL_TRUE, 0, get_b_memsize(), b, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, c_gpu, CL_TRUE, 0, get_c_memsize(), c_start_copy.data(), 0, NULL, NULL);    
    /* 20 Nov 2016 :  remove cl finish */ 
    
  }

  void opencl_memory_release(){
    ret = clReleaseMemObject(c_gpu);
    ret = clReleaseMemObject(a_gpu);
    ret = clReleaseMemObject(b_gpu);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
  }

  /* dev code's placenta to the outside world */  
  void base_benchgemm(std::string kernel_filename, size_t number_of_runs){
    //tinygemm::benchgemm(context, command_queue, device_id_to_use, kernel_filename, number_of_runs, floattostring::get_float_char<TFloat>(), gg, alpha, beta, a_gpu, b_gpu, c_gpu, true, mowri.filename);
    
    tinygemm::benchgemm(command_queue, kernel_filename, number_of_runs, floattostring::get_float_char<TFloat>(), gg, alpha, beta, a_gpu, b_gpu, c_gpu, true, mowri.filename);
  }
  
  tinygemm::TinyGemmSolution nonconst_find(float allotted_time, bool enforce_deterministic){

    tinygemm::TinyGemmSolution tgs = tinygemm::nonconst_find(
      allotted_time,
      //context,
      command_queue,
      //device_id_to_use,
      a_gpu,   
      b_gpu,
      c_gpu,
      enforce_deterministic, 
      floattostring::get_float_char<TFloat>(),  
      gg,
      alpha,
      beta, 
      true, // yes, write to terminal (may be captured further upstream)
      outputfilename); // file where to write the output (if "", nowhere). A new mowri will be created in the function called. 
   
   return tgs;
    
  }

  void base_basegemm_with_accuracy_test(std::string kernel_filename){
    clEnqueueWriteBuffer(command_queue, c_gpu, CL_TRUE, 0, get_c_memsize(), c_start_copy.data(), 0, NULL, NULL);
    /* 20 Nov 2016 : remove cl finish */
    base_benchgemm(kernel_filename, 1);
    ret = clEnqueueReadBuffer(command_queue, c_gpu, CL_TRUE, 0, get_c_memsize(), c, 0, NULL, NULL); 
  }

      
  void benchgemm_gpu(std::vector<std::vector<std::string> > gpu_kernel_filenames, unsigned n_runs, unsigned do_test,  const TFloat * c_true_for_test){
    
    /* A check that the gpu kernel filenames are valid */
    kernelutil::check_gpu_kernel_filenames(gpu_kernel_filenames);
    size_t number_of_kernels_being_tested = gpu_kernel_filenames.size();
    size_t this_run = 0;
    
    
    try{
      for (auto & v : gpu_kernel_filenames){
        ++this_run;
        /* Originally, gemini was designed to allow for option of running several (4) kernels in parallel. */
        /* Later, with the pointer relocation trick (cite TODO), we realised that this was not necessary. */
        /* So here we just take the first filename in the vector */
        std::string filename(v[0]);
        mowri << "\nSource kernel " << "(" << this_run << "/" << number_of_kernels_being_tested << ") " << filename << Endl;      
        base_benchgemm(filename, n_runs);
        if (do_test != 0){
          mowri << "about to perform test in Gemini benchgemm_gpu" << Endl;
          base_basegemm_with_accuracy_test(filename);
        }
        
        /* 20 Nov 2016 : remove cl flush and finish */
        
        if (do_test != 0){
          /* compare */
          accuracytests::accuracy_test(gg.isColMajor, gg.tC, gg.m, gg.n, gg.ldc, c_true_for_test, c, gg.c_offset, mowri, 1e-6);
        }
      }
    }
    
    

      
    catch(...){
      mowri << "caught exception, releasing cl memory in gemini.hpp" << Endl;
      opencl_memory_release();
      throw;
    }
    opencl_memory_release();
  }
};




void hello(){
  std::cout << "hello!" <<std::endl;
}



template <typename TFloat>
void benchgemm(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, TFloat alpha, const TFloat * a, unsigned lda, unsigned a_offset, const TFloat * b, unsigned ldb, unsigned b_offset, TFloat beta, TFloat * c, unsigned ldc, unsigned c_offset, std::vector<std::string> cpu_algs, std::vector<std::vector<std::string> > gpu_kernel_filenames, bool capture_output, std::string & output, const TFloat * c_true_for_test, unsigned do_test, unsigned n_runs, std::string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic){
  
  //if (findfirst == true){
    //tinygemm::TinyGemmGeometry gg(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k);  
    //Gemini <TFloat> gem (gg, a, b, c, alpha, beta, outputfilename);
    //tinygemm::TinyGemmSolution tgs = gem.find(allotted_time); 
    //main_kernel = tgs.main_kernel;
  //}
  
  if (findfirst == true && gpu_kernel_filenames.size() != 0){
    throw tinygemm_error( "findfirst is true, and so gpu_kernel_filenames should be an empty list \n");
    //tinygemm_error("findfirst is true, and so gpu_kernel_filenames should be an empty list \n ");
  }
  
  if (findfirst == true && cpu_algs.size() != 0){
    throw tinygemm_error("findfirst is true, and so cpu_algs should be an empty list \n ");
  }
  
  if (allotted_time >= 0 and findfirst == false){
    throw tinygemm_error("allotted_time is positive, so findfirst was expected to be true. We are being pedantic here, but please set allotted_time to a negative value. Just checking that you're aware that allotted_time is specific to the find algorith");
  }
  
  tinygemm::TinyGemmGeometry gg(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset);  
  std::stringstream buffer;
  auto cout_buff = std::cout.rdbuf();
  if (capture_output == true){
    std::cout.rdbuf(buffer.rdbuf());
  }
  std::ofstream nowhere;
  Gemini <TFloat> gem (gg, a, b, c, alpha, beta, outputfilename);
  
  
  if (findfirst == true){
    tinygemm::TinyGemmSolution tgs = gem.nonconst_find(allotted_time, enforce_deterministic); 
    std::string kernelfilename = defpaths::scratchpadfinddir + "/kernfoundinbenchmark.cl"; 
    std::ofstream out(kernelfilename);
    out << tgs.main_kernel;
    out.close();
    gpu_kernel_filenames = {{ kernelfilename }};
  }
  
  gem.benchgemm_cpu(cpu_algs);
  gem.benchgemm_gpu(gpu_kernel_filenames, n_runs, do_test, c_true_for_test);
  
  if (capture_output == true){
    output = buffer.str();
    std::cout.rdbuf(cout_buff);
  }
  
  else{
    output = "capture_output was false, so nothing here";
  }
}

template void benchgemm(bool isColMajor, bool tA, bool tB, bool tC,  unsigned m, unsigned n, unsigned k, float alpha, const float * a, unsigned lda, unsigned a_offset, const float * b, unsigned ldb, unsigned b_offset, float beta, float * c, unsigned ldc, unsigned c_offset, std::vector<std::string> cpu_algs, std::vector<std::vector<std::string> > gpu_kernel_filenames, bool capture_output, std::string & output, const float * c_true_for_test, unsigned do_test, unsigned n_runs, std::string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic);

template void benchgemm(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, double alpha, const double * a, unsigned lda, unsigned a_offset, const double * b, unsigned ldb, unsigned b_offset, double beta, double * c, unsigned ldc, unsigned c_offset, std::vector<std::string> cpu_algs, std::vector<std::vector<std::string> > gpu_kernel_filenames, bool capture_output, std::string & output, const double * c_true_for_test, unsigned do_test, unsigned n_runs, std::string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic);




}
}

