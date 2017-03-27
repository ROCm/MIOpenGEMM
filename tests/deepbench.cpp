#include "iterexperiments.hpp"


int main(){

  auto geometries = get_deepbench_geometries();
  float allotted_time = 200.00; 
  unsigned n_iterations = 3;
  bool verbose = false;
  std::vector<std::string> v_constraints = {""};
  std::string basedir("/home/james/gemmpaper/data/apiless/dblarge/");
  run_find_experiments(geometries, allotted_time, verbose, v_constraints, n_iterations, basedir);
  
}
 

