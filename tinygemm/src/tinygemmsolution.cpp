#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <tinygemm/tinygemmsolution.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/betackernelutil.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{


TinyGemmSolutionStatistics::TinyGemmSolutionStatistics(float median_benchmark_time_, float median_benchmark_gflops_, float solution_discovery_time_): 
  median_benchmark_time(median_benchmark_time_), median_benchmark_gflops(median_benchmark_gflops_), solution_discovery_time(solution_discovery_time_) {}
    
TinyGemmSolution::TinyGemmSolution(
const std::string & betac_kernel_, 
const std::string & betac_kernel_function_name_,  
const std::string & main_kernel_, 
const std::string &  main_kernel_function_name_, 
const hyperparams::HyperParams & hp_, 
const derivedparams::DerivedParams & dp_, 
const tinygemm::TinyGemmGeometry & geometry_, 
char floattype_, 
TinyGemmSolutionStatistics tgss_): 

betac_kernel(betac_kernel_), 
betac_kernel_function_name(betac_kernel_function_name_), 
main_kernel(main_kernel_), 
main_kernel_function_name(main_kernel_function_name_),  
hp(hp_), 
dp(dp_), 
geometry(geometry_), 
floattype(floattype_), 
statistics(tgss_), 


main_kernel_n_work_groups(dp.n_work_groups),
main_kernel_local_work_size(dp.n_work_items_per_workgroup),
main_kernel_global_work_size(dp.global_work_size),

betac_global_work_size(betac::get_global_work_size(geometry)),
betac_local_work_size(betac::n_work_items_per_group){
  
}
  
std::string TinyGemmSolution::get_networkconfig_string() const{
  return geometry.get_networkconfig_string();
}

std::string TinyGemmSolution::get_hyper_param_string() const{
  return hp.get_string();
}


} //namespace
