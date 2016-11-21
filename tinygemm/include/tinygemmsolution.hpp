#ifndef TINYGEMMSOLUTION_HPP
#define TINYGEMMSOLUTION_HPP


#include <string> 
#include <map>

namespace tinygemm{

class TinyGemmSolutionStatistics{
  public:
    /* the median time and flop/s recorded with the(se) kernel(s) on the specified benchmarked problem */
    float median_benchmark_time;
    float median_benchmark_gflops;
    unsigned benchmarked_m, benchmarked_n, benchmared_k;
    
    /* the time in seconds at which this solution was discovered  */
    float time_when_benchmarked;
  
};

class TinyGemmSolution{

public:
  /* Either an empty string, or the kernel to perform the C <- beta C part of GEMM 
  */
  std::string betac_kernel;


  /* if betac_kernel (above) is empty, this is a full GEMM kernel which performs  
   * C <- C + alpha A*B + beta C. Otherwise, if betac_kernel is not empty, 
   * this is a kernel which just does C <- C + alpha A*B */  
  std::string main_kernel;

  /* The names of the betac and main kernels, ie __kernel void THIS_STRING_HERE */
  std::string betac_kernel_function_name;
  std::string main_kernel_function_name;
                
                  
  /* all hyper parameters (unroll, tile size, etc) and basic geometry parameters (tA, tB, tC, isColMajor)
   * these are the same as the values defined as preprocessor flags in main_kernel.
   * NOTE : these are NOT needed to run kernels, just here to describle what the kernel does under the hood  */
  std::map<std::string, unsigned> allparams;

  /* currently 'f' or 'd' for single and double precision, respectively */
  char floattype;

  TinyGemmSolutionStatistics statistics;


  /* A TinyGemmSolution is only valid for a fixed basic geometry (tA, tB, etc) , but can be used for any size m,n,k, 
   * as long as the kernel macro tile size is not larger than m x n. This function should be used to determine
   * n_work_groups, local_work_size and global_work_size, which are needed when enqueueing main_kernel. See example TODO.
   *   */
  std::map<std::string, size_t> get_main_kernel_worksize_params(unsigned m, unsigned n);
  
  /* ditto the above comment, but for betac_kernel */
  std::map<std::string, size_t> get_betac_kernel_worksize_params(unsigned m, unsigned n);

  /* return a string describing the hyper parameters */
  std::string get_hyper_param_string();

};

}

#endif
