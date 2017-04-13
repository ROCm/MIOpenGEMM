#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <tinygemm/tinygemmsolution.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{


TinyGemmSolutionStatistics::TinyGemmSolutionStatistics(float median_benchmark_time_, float median_benchmark_gflops_, float solution_discovery_time_): 
  median_benchmark_time(median_benchmark_time_), median_benchmark_gflops(median_benchmark_gflops_), solution_discovery_time(solution_discovery_time_) {}

std::string TinyGemmSolution::get_networkconfig_string() const{
  return geometry.get_networkconfig_string();
}

std::string TinyGemmSolution::get_hyper_param_string() const{
  return hyper_param_string;
}


} //namespace
