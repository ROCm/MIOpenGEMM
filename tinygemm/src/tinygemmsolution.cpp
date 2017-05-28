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

std::string TinyGemmSolutionStatistics::get_string() const {
  std::stringstream ss;
  ss << "runtime:" << median_benchmark_time << "  gflops:" << median_benchmark_gflops << "  date:" << date;
  ss << "  (find_params) " << find_params.get_string();
  
  std::string stroo("");
  for (char x : ss.str()){
    if (x != '\n'){
      stroo += x;
    }
  }
  
  return stroo;
}

TinyGemmSolutionStatistics::TinyGemmSolutionStatistics(float median_benchmark_time_, float median_benchmark_gflops_, float solution_discovery_time_, std::string date_, const tinygemm::FindParams & find_params_): 
  median_benchmark_time(median_benchmark_time_), median_benchmark_gflops(median_benchmark_gflops_), solution_discovery_time(solution_discovery_time_),  find_params(find_params_) {
    date = date_;
    if (date.size() > 1){
      if (date[date.size() - 1] == '\n'){
       date.resize(date.size() - 1);
      }
    }    
}

TinyGemmSolutionStatistics::TinyGemmSolutionStatistics(std::string cache_string){
  auto megafrags = stringutil::split(cache_string, "__");

  if (megafrags.size() < 3){
    throw  tinygemm_error("problem constructing tinygemm solution stats from string");
  }
  
  auto get_X = [&megafrags](unsigned i){
    auto X = stringutil::split(megafrags[i]);
    if (X.size() != 2)  {
      throw  tinygemm_error("split not into 2, problem in generating tinygemm solution from string");
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





std::string TinyGemmSolution::get_cache_entry_string(std::string k_comment) const{
  std::stringstream cache_write_ss;
  cache_write_ss << "add_entry(kc, \"" << devinfo.identifier << "\", /* device key */\n";
  cache_write_ss << "\"" << constraints_string << "\", /* constraint key */\n";
  cache_write_ss << "\"" << geometry.get_string() << "\", /* geometry key */\n";
  cache_write_ss << "\"" << k_comment << "\", /* comment key */\n";
  cache_write_ss << "{\"" << hyper_param_string << "\", /* solution hyper string */\n";
  cache_write_ss << "{" << statistics.median_benchmark_time << ", " << statistics.median_benchmark_gflops << ", " << statistics.solution_discovery_time;
  cache_write_ss << ", \"" << statistics.date << "\"" << ", /* solution stats (time [ms], gflops, time found (within descent), date found */\n";
  cache_write_ss << "{" << statistics.find_params.allotted_time <<", " << statistics.find_params.allotted_descents << ", " << statistics.find_params.n_runs_per_kernel << ", " << tinygemm::get_sumstatkey(statistics.find_params.sumstat) << "}}}); /* find param: allotted time, allotted descents, n runs per kernel, summmary over runs */\n\n";  
  return cache_write_ss.str();

}


} //namespace
