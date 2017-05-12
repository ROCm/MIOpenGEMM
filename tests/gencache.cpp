#include "iterexperiments.hpp"

int main(){
  


 /*                      m ,  n ,  k , lda ,ldb ,ldc , tA , tB  */ 
  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>> geotuples = {
    std::make_tuple(5000, 5000, 5000, 5000, 5000, 5000, false, false), 
    std::make_tuple(2000, 2001, 2002, 2003, 2004, 2005, false, false),     
  };
  
  unsigned ws_size = 1;
    
  auto geometries = get_from_m_n_k_ldaABC_tA_tB(geotuples, ws_size);    

  float allotted_time = 100.00; 
  unsigned allotted_iterations = 1;
  unsigned n_runs_per_kernel = 3;
  tinygemm::SummaryStat sumstat(tinygemm::Max);
  bool verbose = true;
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};

  std::string basedir("");
  run_find_experiments(geometries, allotted_time, allotted_iterations, n_runs_per_kernel, sumstat, verbose, v_constraints, basedir);

  

}
