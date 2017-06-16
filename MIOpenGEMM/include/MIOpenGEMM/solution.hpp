#ifndef SOLUTION_HPP
#define SOLUTION_HPP


#include <string> 
#include <map>
#include <chrono>

#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/hyperparams.hpp>
#include <MIOpenGEMM/derivedparams.hpp>
#include <MIOpenGEMM/kernelstring.hpp>
#include <MIOpenGEMM/findparams.hpp>

namespace MIOpenGEMM{

class SolutionStatistics{
  public:
    /* the median time and flop/s recorded with the(se) kernel(s) on the specified benchmarked problem */
    float median_benchmark_time;
    float median_benchmark_gflops;

    /* the time in seconds at which this solution was discovered  (from start of descent)  */
    float solution_discovery_time;
    
    /* timestamp (date) when found */
    std::string date;
    
    FindParams find_params;
    
    SolutionStatistics(float median_benchmark_time, float median_benchmark_gflops, float solution_discovery_time, std::string date, const FindParams & find_params);
    SolutionStatistics(std::string cache_string);

    SolutionStatistics() = default;
    
    std::string get_string() const;
    
    


};

/* Note 01 feb 2017: A Solution is only valid for a fixed geometry */
class Solution{

public:

  /* the geometry on which this solution was obtained */
  Geometry geometry;

  SolutionStatistics statistics;
  
  /* the kernels of which the solution is comprised */
  std::vector<KernelString> v_tgks;

  std::string hyper_param_string;
  
  openclutil::OpenCLDeviceInfo devinfo;

  std::string constraints_string;
  
  Solution(const Geometry & geometry_, SolutionStatistics tgss_, const std::vector<KernelString> & v_tgks_, std::string hyper_param_string_, openclutil::OpenCLDeviceInfo devinfo_, std::string constraints_string_): geometry(geometry_), statistics(tgss_), v_tgks(v_tgks_), hyper_param_string(hyper_param_string_), devinfo(devinfo_), constraints_string(constraints_string_) {}

  /* return a string summarising the Geometry, less offsets (a request from MLOpen) */
  std::string get_networkconfig_string() const;

  /* return a string describing the hyper parameters */
  std::string get_hyper_param_string() const;

  std::string get_cache_entry_string(std::string k_comment = "") const;
  

};

}

#endif
