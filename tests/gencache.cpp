#include <tinygemm/iterexperiments.hpp>
#include <tinygemm/tggeometryutil.hpp>

int main(){
  


  /* (1) define a vector of geometries which you wish to generate cachfe entries for */
  /*  m ,  n ,  k , lda ,ldb ,ldc , tA , tB  */ 
  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>> geotuples = {
    //std::make_tuple(1000,1000,1000,1000,1000,1000, false, false), 
    std::make_tuple(100,200,300,400,500,600, false, false),     
    std::make_tuple(101,200,300,400,500,600, false, false),         
  };
  unsigned ws_size = 1;
  auto geometries = tinygemm::get_from_m_n_k_ldaABC_tA_tB(geotuples, ws_size);    

  /* define how long you want to search for */
  /* the maximum time to search for */
  float allotted_time = 1.00; 
  /* the maximum number of restarts */
  unsigned allotted_iterations = 100;
  /* the number of times each kernel should be run during the search. tradeoff : many runs means less exploration with more accurate estimates */
  unsigned n_runs_per_kernel = 3;
  /* the statistic for averaging over the n_runs_per_kernel runs. Max/Mean/Median */  
  tinygemm::SummaryStat sumstat(tinygemm::Max);
  tinygemm::FindParams find_params(allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);

  
  bool verbose = false;
  std::string basedir("/home/james/tinygemmout/test1"); /* path to a directory if you want a log of each of the searches (recommended) */
    
  /* the constraints on the kernel. "A_WOS0__B_WOS0" is for no workspace in GEMM. "A_WOS0__B_WOS0__C_ICE_1" is for deterministic  */
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};


  bool verbose_outer = true;
  std::string fn_outer("");
  
  tinygemm::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);


  

}
