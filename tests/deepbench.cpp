#include "iterexperiments.hpp"


int main(){

  auto geometries = get_deepbench_geometries();
  float allotted_time = 15.00; 
  unsigned allotted_iterations = 5;
  unsigned n_runs_per_kernel = 1;
  tinygemm::SummaryStat sumstat(tinygemm::Max);
  bool verbose = true;
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};
  std::string basedir("/home/james/tinygemmout/test1");
  run_find_experiments(geometries, allotted_time, allotted_iterations, n_runs_per_kernel, sumstat, verbose, v_constraints, basedir);
  
}
 

