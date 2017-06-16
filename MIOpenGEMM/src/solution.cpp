#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <string>

#include <MIOpenGEMM/solution.hpp>
#include <MIOpenGEMM/sizingup.hpp>
#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/stringutilbase.hpp>

namespace MIOpenGEMM{

std::string SolutionStatistics::get_string() const {
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

SolutionStatistics::SolutionStatistics(float median_benchmark_time_, float median_benchmark_gflops_, float solution_discovery_time_, std::string date_, const FindParams & find_params_): 
  median_benchmark_time(median_benchmark_time_), median_benchmark_gflops(median_benchmark_gflops_), solution_discovery_time(solution_discovery_time_),  find_params(find_params_) {
    date = date_;
    if (date.size() > 1){
      if (date[date.size() - 1] == '\n'){
       date.resize(date.size() - 1);
      }
    }    
}

SolutionStatistics::SolutionStatistics(std::string cache_string){
  auto megafrags = stringutil::split(cache_string, "__");

  if (megafrags.size() < 3){
    throw  miog_error("problem constructing solution stats from string, in constructor");
  }
  
  auto get_X = [&megafrags](unsigned i){
    auto X = stringutil::split(megafrags[i]);
    if (X.size() != 2)  {
      throw  miog_error("split not into 2, problem in generating solution from string, on constructor");
    }
    return X[1];
  };
  
  median_benchmark_time = std::stof(get_X(0));
  median_benchmark_gflops = std::stof(get_X(1));
  solution_discovery_time = 0;
  date = get_X(2);
}
    
std::string Solution::get_networkconfig_string() const{
  return geometry.get_networkconfig_string();
}

std::string Solution::get_hyper_param_string() const{
  return hyper_param_string;
}





std::string Solution::get_cache_entry_string(std::string k_comment) const{
  std::stringstream cache_write_ss;
  cache_write_ss << "add_entry(kc, \"" << devinfo.identifier << "\", /* device key */\n";
  cache_write_ss << "\"" << constraints_string << "\", /* constraint key */\n";
  cache_write_ss << "\"" << geometry.get_string() << "\", /* geometry key */\n";
  cache_write_ss << "\"" << k_comment << "\", /* comment key */\n";
  cache_write_ss << "{\"" << hyper_param_string << "\", /* solution hyper string */\n";
  cache_write_ss << "{" << statistics.median_benchmark_time << ", " << statistics.median_benchmark_gflops << ", " << statistics.solution_discovery_time;
  cache_write_ss << ", \"" << statistics.date << "\"" << ", /* solution stats (time [ms], gflops, time found (within descent), date found */\n";
  cache_write_ss << "{" << statistics.find_params.allotted_time <<", " << statistics.find_params.allotted_descents << ", " << statistics.find_params.n_runs_per_kernel << ", " << get_sumstatkey(statistics.find_params.sumstat) << "}}}); /* find param: allotted time, allotted descents, n runs per kernel, summmary over runs */\n\n";  
  return cache_write_ss.str();

}


} //namespace
