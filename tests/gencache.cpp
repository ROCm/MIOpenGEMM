#include <MIOpenGEMM/iterexperiments.hpp>
#include <MIOpenGEMM/tggeometryutil.hpp>

/* example of how to generate cache entries */
int main(){
    
  


  /* (1)   define a vector of geometries which you wish to generate cache entries for. */
  /* (1.1) This can be done with strings  */
  std::vector<MIOpenGEMM::Geometry> geometries = {{"tC0_tA1_tB0_colMaj1_m1200_n1_k1_lda1_ldb1_ldc1200_ws0_f32", }};
  
  
  if (false){
    /* (1.2) Or it can be done `directly' */
    geometries.emplace_back(
    /*isColMajor tA tB tC lda ldb ldc m n k workspace_size floattype */
        true, false, false, false, 10, 10, 10, 10, 10, 10, 0,'f');
   
    /* (1.3) or using a function which assumes tC is false and isColMajor is true  (m, n, k, lda, ldb, ldc, tA, tB), workspace_size */ 
    auto more_geometries = MIOpenGEMM::get_from_m_n_k_ldaABC_tA_tB({std::make_tuple(800, 64, 16, 16, 16, 800, true, false)}, 0);
    geometries.insert(geometries.end(), more_geometries.begin(), more_geometries.end());
    
    /* (1.4) or using a function which also assumes `minimal' ldA, ldB, ldC */
    more_geometries = MIOpenGEMM::get_from_m_n_k_tA_tB({std::make_tuple(20, 30, 40, false, false)}, 0);
    geometries.insert(geometries.end(), more_geometries.begin(), more_geometries.end());
  }


    
  /* (2)   define the search settings (upper bounds for each of the geometries) */
  /* the maximum time to search, per geometry */
  float allotted_time = 120.00; 
  /* the maximum number of restarts during the search, per geometry */
  unsigned allotted_iterations = 30;
  /* the number of times each kernel should be run during the search. (tradeoff : many runs means less exploration with more accurate perf estimates) */
  unsigned n_runs_per_kernel = 3;
  /* the statistic for averaging over the n_runs_per_kernel runs. Max/Mean/Median (TODO. currently only Max supported) */  
  MIOpenGEMM::SummaryStat sumstat(MIOpenGEMM::Max);
  MIOpenGEMM::FindParams find_params(allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);

  
  bool verbose = false;
  /* path to a directory if you want a log of each of the searches (not nec, but useful for further analysis/debugging) */
  std::string basedir("/home/james/miog_out/test1");
    
  /* the constraints on the kernel. 
   * "A_WOS0__B_WOS0" is for no workspace in GEMM 
   * "A_WOS0__B_WOS0__C_ICE_1" is for no workspace and deterministic  */
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};


  bool verbose_outer = true;
  std::string fn_outer("");
  MIOpenGEMM::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);

}
