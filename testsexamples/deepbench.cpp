#include "basicfind.hpp"
#include <sstream>
#include <chrono>

int main(){

  std::vector<std::tuple<int, int, int, bool, bool>> problems = {
    /* taken from https://github.com/baidu-research/DeepBench/blob/master/code/nvidia/gemm_bench.cu */
    std::make_tuple(1760, 16, 1760, false, false),
    std::make_tuple(1760, 32, 1760, false, false),
    std::make_tuple(1760, 64, 1760, false, false),
    std::make_tuple(1760, 128, 1760, false, false),
    std::make_tuple(1760, 7000, 1760, false, false),
    std::make_tuple(2048, 16, 2048, false, false),
    std::make_tuple(2048, 32, 2048, false, false),
    std::make_tuple(2048, 64, 2048, false, false),
    std::make_tuple(2048, 128, 2048, false, false),
    std::make_tuple(2048, 7000, 2048, false, false),
    std::make_tuple(2560, 16, 2560, false, false),
    std::make_tuple(2560, 32, 2560, false, false),
    std::make_tuple(2560, 64, 2560, false, false),
    std::make_tuple(2560, 128, 2560, false, false),
    std::make_tuple(2560, 7000, 2560, false, false),
    std::make_tuple(4096, 16, 4096, false, false),
    std::make_tuple(4096, 32, 4096, false, false),
    std::make_tuple(4096, 64, 4096, false, false),
    std::make_tuple(4096, 128, 4096, false, false),
    std::make_tuple(4096, 7000, 4096, false, false),
    std::make_tuple(1760, 7133, 1760, false, true),
    std::make_tuple(2048, 7133, 2048, false, true),
    std::make_tuple(2560, 7133, 2560, false, true),
    std::make_tuple(4096, 7133, 4096, false, true),
    std::make_tuple(5124, 9124, 1760, false, false),
    std::make_tuple(35, 8457, 1760, false, false),
    std::make_tuple(5124, 9124, 2048, false, false),
    std::make_tuple(35, 8457, 2048, false, false),
    std::make_tuple(5124, 9124, 2560, false, false),
    std::make_tuple(35, 8457, 2560, false, false),
    std::make_tuple(5124, 9124, 4096, false, false),
    std::make_tuple(35, 8457, 4096, false, false),
    std::make_tuple(7680, 16, 2560, false, false),
    std::make_tuple(7680, 32, 2560, false, false),
    std::make_tuple(7680, 64, 2560, false, false),
    std::make_tuple(7680, 128, 2560, false, false),
    std::make_tuple(3072, 16, 1024, false, false),
    std::make_tuple(3072, 32, 1024, false, false),
    std::make_tuple(3072, 64, 1024, false, false),
    std::make_tuple(3072, 128, 1024, false, false),
    std::make_tuple(3072, 7435, 1024, false, true),
    std::make_tuple(7680, 5481, 2560, false, true),
    
    /* dodgey, but m = k so it works */
    std::make_tuple(2048, 16, 2048, true, false),
    std::make_tuple(2048, 32, 2048, true, false),
    std::make_tuple(2048, 64, 2048, true, false),
    std::make_tuple(2048, 128, 2048, true, false),
    std::make_tuple(2048, 7000, 2048, true, false),
    std::make_tuple(2560, 16, 2560, true, false),
    std::make_tuple(2560, 32, 2560, true, false),
    std::make_tuple(2560, 64, 2560, true, false),
    std::make_tuple(2560, 128, 2560, true, false),
    std::make_tuple(2560, 7000, 2560, true, false),
    std::make_tuple(4096, 16, 4096, true, false),
    std::make_tuple(4096, 32, 4096, true, false),
    std::make_tuple(4096, 64, 4096, true, false),
    std::make_tuple(4096, 128, 4096, true, false),
    std::make_tuple(4096, 7000, 4096, true, false),
    std::make_tuple(1760, 16, 1760, true, false),
    std::make_tuple(1760, 32, 1760, true, false),
    std::make_tuple(1760, 64, 1760, true, false),
    std::make_tuple(1760, 128, 1760, true, false),
    std::make_tuple(1760, 7000, 1760, true, false),
    
    
    /* The dodgey cases */
    std::make_tuple(7680, 16, 2560, true, false),
    std::make_tuple(7680, 32, 2560, true, false),
    std::make_tuple(7680, 64, 2560, true, false),
    std::make_tuple(7680, 128, 2560, true, false),
    std::make_tuple(5124, 9124, 1760, true, false),
    std::make_tuple(35, 8457, 1760, true, false),
    std::make_tuple(5124, 9124, 2048, true, false),
    std::make_tuple(35, 8457, 2048, true, false),
    std::make_tuple(5124, 9124, 2560, true, false),
    std::make_tuple(35, 8457, 2560, true, false),
    std::make_tuple(5124, 9124, 4096, true, false),
    std::make_tuple(35, 8457, 4096, true, false),
    std::make_tuple(3072, 16, 1024, true, false),
    std::make_tuple(3072, 32, 1024, true, false),
    std::make_tuple(3072, 64, 1024, true, false),
    std::make_tuple(3072, 128, 1024, true, false),
    
    
    /* dodgeys fixed */
    std::make_tuple(2560, 16, 7680, true, false),
    std::make_tuple(2560, 32, 7680, true, false),
    std::make_tuple(2560, 64, 7680, true, false),
    std::make_tuple(2560, 128, 7680, true, false),
    std::make_tuple(1760, 9124, 5124, true, false),
    std::make_tuple(1760, 8457, 35, true, false),
    std::make_tuple(2048, 9124, 5124, true, false),
    std::make_tuple(2048, 8457, 35, true, false),
    std::make_tuple(2560, 9124, 5124, true, false),
    std::make_tuple(2560, 8457, 35, true, false),
    std::make_tuple(4096, 9124, 5124, true, false),
    std::make_tuple(4096, 8457, 35, true, false),
    std::make_tuple(1024, 16, 3072, true, false),
    std::make_tuple(1024, 32, 3072, true, false),
    std::make_tuple(1024, 64, 3072, true, false),
    std::make_tuple(1024, 128, 3072, true, false)
  
  };
  
  bool isColMajor = true;
  bool tC = false;
  //double alpha = 1.43235342345;
  //double beta = 0.45348379373;
  float allotted_time = 1.003; 
  bool verbose = false;
  
  /* enforce_deterministc, ldx_offset */
  std::vector<std::tuple<bool, unsigned>> run_settings = {
    std::make_tuple(false,1), 
    std::make_tuple(false,0), 
  }; 
  
  /* We're just tracking the overall run time with these */
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms = end - start;
  float elapsed_seconds = fp_ms.count();
  
  
  
  for (const auto & run_setting :  run_settings){
  
    bool enforce_deterministic;
    unsigned ldx_offset;
    std::tie(enforce_deterministic, ldx_offset) = run_setting;
    
    std::cout << "\n\n\nEntering benchmarking experiment with enforce_deterministic = " << enforce_deterministic << " and ldx_offset = " << ldx_offset << std::endl;
    
    
    
    //for (unsigned prob_i = 0; prob_i < problems.size(); ++prob_i){
    
   for (unsigned prob_i = 0; prob_i < 2; ++prob_i){
    
      auto problem = problems[prob_i];
      int m, n, k;
      bool tA, tB;
      std::tie(m, n, k, tA, tB) = problem;
      
      end = std::chrono::high_resolution_clock::now();
      fp_ms = end - start;
      elapsed_seconds = fp_ms.count();
      
      std::cout << (prob_i + 1) <<  "/" <<  problems.size() << " \t m:" << m << " \t n:" << n << " \t k:" << k << " \t tA:" << tA << " \t tB:" << m << "  \t  elapsed time : " << elapsed_seconds << " [s]" << std::endl;    
      
      
      std::stringstream ss_logfile;
#ifdef DIR_FOR_WRITING
      ss_logfile << DIR_FOR_WRITING << "/deepbench/" << "at" << int(allotted_time) << "_off" << ldx_offset << "_ed" << int(enforce_deterministic) << "_m" << m  << "_n" << n  << "_k" << k  << "_tA" << tA  << "_tB" << tB << ".txt";   
#endif
      
      unsigned lda = (tA == isColMajor ? k : m) + (ldx_offset == 1 ? 5 : 0);
      unsigned ldb = (tB == isColMajor ? n : k) + (ldx_offset == 1 ? 7 : 0);
      unsigned ldc = (tC == isColMajor ? n : m) + (ldx_offset == 1 ? 13 : 0);
      
      unsigned a_offset = 0;
      unsigned b_offset = 0;
      unsigned c_offset = 0;
 
 
      unsigned tail_off_a = 0;
      unsigned tail_off_b = 0;
      unsigned tail_off_c = 0;

      
      unsigned n_postfind_runs = 11;
      bool do_cpu_test = false;
      
      unsigned workspace_size = 3;
      unsigned workspace_offset = 4;      
  

      char floattype = 'f';
      tinygemm::TinyGemmGeometry gg (isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
      tinygemm::TinyGemmOffsets offsets (a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c);

      basicfind<float>(gg, offsets, allotted_time, verbose, ss_logfile.str(), enforce_deterministic, n_postfind_runs, do_cpu_test);    
    }
  }
  
  return 0;
}



