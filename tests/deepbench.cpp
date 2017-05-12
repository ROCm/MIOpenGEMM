#include <tinygemm/iterexperiments.hpp>


int main(){

  auto geometries = tinygemm::get_deepbench_geometries();
  float allotted_time = 3.00; 
  unsigned allotted_iterations = 2;
  unsigned n_runs_per_kernel = 1;
  tinygemm::SummaryStat sumstat(tinygemm::Max);
  bool verbose = true;
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};
  std::string basedir("/home/james/tinygemmout/test1");
  
  tinygemm::FindParams find_params(allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);
  
  
  tinygemm::run_find_experiments(geometries, find_params, verbose, v_constraints, basedir);
  

  return 0;
}
 

