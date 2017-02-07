#ifndef TINYGEMMSOLUTION_HPP
#define TINYGEMMSOLUTION_HPP


#include <string> 
#include <map>


#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/kernelstring.hpp>

namespace tinygemm{

class TinyGemmSolutionStatistics{
  public:
    /* the median time and flop/s recorded with the(se) kernel(s) on the specified benchmarked problem */
    float median_benchmark_time;
    float median_benchmark_gflops;

    /* the time in seconds at which this solution was discovered  */
    float solution_discovery_time;
    
    TinyGemmSolutionStatistics(float median_benchmark_time, float median_benchmark_gflops, float solution_discovery_time);
     
};

/* Note 01 feb 2017: A TinyGemmSolution is only valid for a fixed geometry */
class TinyGemmSolution{

public:

  /* all hyper-parameters (unroll, tile size, etc)
   * these are also defined as preprocessor flags in kernel strings
   * NOTE : these are NOT needed to run kernels, 
   * just here to describle what the kernel does under the hood  */
  hyperparams::HyperParams hp;

  /* the geometry on which this solution was obtained */
  tinygemm::TinyGemmGeometry geometry;

  /* all derived parameters (derived from hyper-parameters and geometry) */
  derivedparams::DerivedParams dp;
  
  TinyGemmSolutionStatistics statistics;

  /* the kernels of which the solution is comprised */
  std::vector<KernelString> v_tgks;

 TinyGemmSolution(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & geometry_, const derivedparams::DerivedParams & dp_, TinyGemmSolutionStatistics tgss_, const std::vector<KernelString> & v_tgks_): hp(hp_), geometry(geometry_), dp(dp_), statistics(tgss_), v_tgks(v_tgks_) {}

  /* return a string summarising the TinyGemmGeometry, less offsets (a request from MLOpen) */
  std::string get_networkconfig_string() const;

  /* return a string describing the hyper parameters */
  std::string get_hyper_param_string() const;
  

};

}

#endif



  ///* Either an empty string, or the kernel to perform the C <- beta C part of GEMM */
  //std::string betac_kernel;

  ///* The name of the betac kernel, ie __kernel void THIS_STRING_HERE */
  //std::string betac_kernel_function_name;

  //const size_t betac_local_work_size;
  //const size_t betac_global_work_size;



  ///* if betac_kernel (above) is empty, this is a full GEMM kernel which performs  
   //* C <- C + alpha A*B + beta C. Otherwise, if betac_kernel is not empty, 
   //* this is a kernel which just does C <- C + alpha A*B */  
  //std::string main_kernel;

  ///* The name of the main kernel, ie __kernel void THIS_STRING_HERE */
  //std::string main_kernel_function_name;

  //const size_t main_kernel_local_work_size;
  //const size_t main_kernel_global_work_size;




  ///* TODO : move betac_kernel and main_kernel into constructor */
  //TinyGemmSolution(const std::string & betac_kernel, const std::string & betac_kernel_function_name,  const std::string &  main_kernel, const std::string & main_kernel_function_name, const hyperparams::HyperParams & hp, const derivedparams::DerivedParams & dp, const tinygemm::TinyGemmGeometry & geometry, TinyGemmSolutionStatistics tgss);
