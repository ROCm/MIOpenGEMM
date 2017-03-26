#include "basicfind.hpp"
#include <sstream>
#include <chrono>

std::string get_padded(unsigned x, unsigned length = 4){
  auto n_pads = length + 1 - unsigned(std::log10(x + 1.));
  std::string padded = std::to_string(x);
  for (auto sp = 0; sp < n_pads; ++sp){
    padded = padded + " ";
  }
  return padded;
}


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
    
    
    /////* The dodgey cases */
    //std::make_tuple(7680, 16, 2560, true, false),
    //std::make_tuple(7680, 32, 2560, true, false),
    //std::make_tuple(7680, 64, 2560, true, false),
    //std::make_tuple(7680, 128, 2560, true, false),
    //std::make_tuple(5124, 9124, 1760, true, false),
    //std::make_tuple(35, 8457, 1760, true, false),
    //std::make_tuple(5124, 9124, 2048, true, false),
    //std::make_tuple(35, 8457, 2048, true, false),
    //std::make_tuple(5124, 9124, 2560, true, false),
    //std::make_tuple(35, 8457, 2560, true, false),
    //std::make_tuple(5124, 9124, 4096, true, false),
    //std::make_tuple(35, 8457, 4096, true, false),
    //std::make_tuple(3072, 16, 1024, true, false),
    //std::make_tuple(3072, 32, 1024, true, false),
    //std::make_tuple(3072, 64, 1024, true, false),
    //std::make_tuple(3072, 128, 1024, true, false),
    
    
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
  float allotted_time = 200.00; 
  bool verbose = false;
  
  /* constraint_string, ldx_offset */
  std::vector<std::tuple<std::string, unsigned>> run_settings = {
    std::make_tuple("",0),
    std::make_tuple("C_ICE1",0),
    std::make_tuple("C_UFO0",0),
    std::make_tuple("A_LIW0__B_LIW0",0), 
  };
    
  /* We're tracking the overall run time with these */
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms = end - start;
  float elapsed_seconds = fp_ms.count();

  unsigned n_iterations = 30;
  for (unsigned iteration = 0; iteration < n_iterations; ++iteration){
    std::cout << "\n\nEntering iteration " << iteration <<  " of "  << n_iterations << std::endl;
  
    for (const auto & run_setting :  run_settings){
      std::string constraint_string;
      unsigned ldx_offset;
      std::tie(constraint_string, ldx_offset) = run_setting;
      std::cout << "\nEntering benchmarking experiment with constraint_string = (" << constraint_string << ") and ldx_offset = " << ldx_offset << std::endl;
      std::string dir_for_writing("/home/james/gemmpaper/data/apiless/dbsmall/");
      std::stringstream fulldir_ss;
      fulldir_ss << dir_for_writing  << "deepbench" << iteration << "/";
      std::string fulldir = fulldir_ss.str();
      std::string syscall = std::string("mkdir -p  ") + fulldir;
      std::system(syscall.c_str());

      for (unsigned prob_i = 0; prob_i < problems.size(); ++prob_i){
        
        auto problem = problems[prob_i];
        int m, n, k;
        bool tA, tB;
        std::tie(m, n, k, tA, tB) = problem;
        unsigned area_threshold = 100000;
        if (m*n > 100000){
          std::cout << "skipping the problem as m*n > " << area_threshold << std::endl;
        }
        
        else {
        
          std::stringstream ss_logfile;        
          ss_logfile << fulldir << "/" << "at" << int(allotted_time) << "_off" << ldx_offset << "_cs" << constraint_string << "_m" << m  << "_n" << n  << "_k" << k  << "_tA" << tA  << "_tB" << tB << ".txt";
          
          unsigned lda = (tA == isColMajor ? k : m) + (ldx_offset == 1 ? 5 : 0);
          unsigned ldb = (tB == isColMajor ? n : k) + (ldx_offset == 1 ? 7 : 0);
          unsigned ldc = (tC == isColMajor ? n : m) + (ldx_offset == 1 ? 13 : 0);
          
          unsigned a_offset = 0;
          unsigned b_offset = 0;
          unsigned c_offset = 0;
     
     
          unsigned tail_off_a = 0;
          unsigned tail_off_b = 0;
          unsigned tail_off_c = 0;
    
          
          unsigned n_postfind_runs = 0;
          bool do_cpu_test = false;
          
          unsigned workspace_size = 3;
          unsigned workspace_offset = 4;      
      
          tinygemm::FindStartType fst(tinygemm::FindStartType::Random);
    
          char floattype = 'f';
          tinygemm::TinyGemmGeometry gg (isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
          tinygemm::TinyGemmOffsets offsets (a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c);
    
          auto soln = basicfind<float>(gg, offsets, allotted_time, verbose, ss_logfile.str(), constraint_string, fst,  n_postfind_runs, do_cpu_test);    
  
          end = std::chrono::high_resolution_clock::now();
          fp_ms = end - start;
          elapsed_seconds = fp_ms.count();
  
          std::cout << (prob_i + 1) <<  "/" <<  problems.size() << " \t m:" << get_padded(m) << " \t n:" << get_padded(n) << " \t k:" << get_padded(k) << " \t tA:" << tA << " \t tB:" << tB << " \tsoln median gflops :  " << soln.statistics.median_benchmark_gflops << "  \t soln median time : " << soln.statistics.median_benchmark_time << "  \t  elapsed time : " << elapsed_seconds << " [s] " << std::endl;

        }
      }
    }
  }
  
  return 0;
}



