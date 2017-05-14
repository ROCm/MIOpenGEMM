#include <tinygemm/iterexperiments.hpp>
#include <tinygemm/tggeometryutil.hpp>

int main(){
  


 /*                      m ,  n ,  k , lda ,ldb ,ldc , tA , tB  */ 
  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>> geotuples = {
    std::make_tuple(5000, 5000, 5000, 5000, 5000, 5000, false, false), 
    std::make_tuple(2000, 2001, 2002, 2003, 2004, 2005, false, false),     
  };
  
  unsigned ws_size = 1;
    
  auto geometries = tinygemm::get_from_m_n_k_ldaABC_tA_tB(geotuples, ws_size);    

  float allotted_time = 3.00; 
  unsigned allotted_iterations = 2;
  unsigned n_runs_per_kernel = 1;
  tinygemm::SummaryStat sumstat(tinygemm::Max);
  bool verbose = false;
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};
  std::string basedir("/home/james/tinygemmout/test1");
  
  tinygemm::FindParams find_params(allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);
  
 
  bool verbose_outer = true;
  std::string fn_outer("");
  tinygemm::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);


  

}
