#include "iterexperiments.hpp"

int main(){
  
  std::vector<tinygemm::TinyGemmGeometry> geometries = get_deepbench_geometries();
  float allotted_time = 200.00; 
  unsigned n_iterations = 10;
  bool verbose = false;
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};  
  

}
