#include <MIOpenGEMM/iterexperiments.hpp>


int main(){

  auto geometries = MIOpenGEMM::get_deepbench_geometries();
  float allotted_time = 3.00; 
  unsigned allotted_iterations = 2;
  unsigned n_runs_per_kernel = 1;
  MIOpenGEMM::SummaryStat sumstat(MIOpenGEMM::Max);

  bool verbose = false;
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};
  std::string basedir("/home/james/miogout/test1");
  
  MIOpenGEMM::FindParams find_params(allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);
  
 
  bool verbose_outer = true;
  std::string fn_outer("");
  MIOpenGEMM::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);
  

  return 0;
}
 

