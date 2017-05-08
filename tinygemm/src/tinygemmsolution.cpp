#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <string>

#include <tinygemm/tinygemmsolution.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/stringutilbase.hpp>

namespace tinygemm{


TinyGemmSolutionStatistics::TinyGemmSolutionStatistics(float median_benchmark_time_, float median_benchmark_gflops_, float solution_discovery_time_, std::string date_): 
  median_benchmark_time(median_benchmark_time_), median_benchmark_gflops(median_benchmark_gflops_), solution_discovery_time(solution_discovery_time_), date(date_) {}

TinyGemmSolutionStatistics::TinyGemmSolutionStatistics(std::string cache_string){
  auto megafrags = stringutil::split(cache_string, "__");

  
  if (megafrags.size() < 3){
    throw tinygemm_error("problem constructing tinygemm solution stats from string");
  }
  
  auto get_X = [&megafrags](unsigned i){
    auto X = stringutil::split(megafrags[i]);
    if (X.size() != 2)  {
      throw tinygemm_error("split not into 2, problem in generating tinygemm solution from string");
    }
    return X[1];
  };
  
  
  median_benchmark_time = std::stof(get_X(0));
  median_benchmark_gflops = std::stof(get_X(1));
  solution_discovery_time = 0;
  date = get_X(2);
}
    
std::string TinyGemmSolution::get_networkconfig_string() const{
  return geometry.get_networkconfig_string();
}

std::string TinyGemmSolution::get_hyper_param_string() const{
  return hyper_param_string;
}


} //namespace
