#include <tinygemm/iterexperiments.hpp>
#include <tinygemm/tggeometryutil.hpp>

int main(){
  


  /* define a vector of geometries which you wish to generate cache entries for */
  /*  m ,  n ,  k , lda ,ldb ,ldc , tA , tB  (here we assume tC = false isColMajor = true)  */ 
  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>> geotuples = {
    /* miopen test configs (im2col+gemm) : */     
    std::make_tuple(800, 64, 16, 16, 16, 800, true, false), /* tC0_tA1_tB0_colMaj1_m800_n64_k16_lda16_ldb16_ldc800_ws0_f32 */
    std::make_tuple(16, 64, 800, 16, 800, 16, false, false),     
    /* etc etc ... */
    std::make_tuple(2,3,5,7,11,13, true, true),         
  };
  /* no workspace gemm kernels : */
  unsigned ws_size = 0;
  std::vector<tinygemm::TinyGemmGeometry> geometries = tinygemm::get_from_m_n_k_ldaABC_tA_tB(geotuples, ws_size);    
  /* for `minimal' ldA, ldB and ldC, consider function get_from_m_n_k_tA_tB */
  /* for different tC and isColMajor, consider using the TinyGemmGeometry constructor directly  */
    
  /* define how long you want to search for (upper bound for each geometry) */
  /* the maximum time */
  float allotted_time = 120.00; 
  /* the maximum number of restarts */
  unsigned allotted_iterations = 30;
  /* the number of times each kernel should be run during the search. (tradeoff : many runs means less exploration with more accurate estimates) */
  unsigned n_runs_per_kernel = 3;
  /* the statistic for averaging over the n_runs_per_kernel runs. Max/Mean/Median (TODO. currently only Max supported) */  
  tinygemm::SummaryStat sumstat(tinygemm::Max);
  tinygemm::FindParams find_params(allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);

  
  bool verbose = false;
  /* path to a directory if you want a log of each of the searches (not nec, but useful for further analysis/debugging) */
  std::string basedir(""); //"/home/james/tinygemmout/test1"
    
  /* the constraints on the kernel. "A_WOS0__B_WOS0" is for no workspace in GEMM. "A_WOS0__B_WOS0__C_ICE_1" is for deterministic  */
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};


  bool verbose_outer = true;
  std::string fn_outer("");
  
  tinygemm::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);


  

}
