#include "basicfind.hpp"
#include <sstream>
#include <chrono>

int main(){
  
  /*                      m ,  n ,  k , lda ,ldb ,ldc , tA , tB (from Mayank, 25/11/2016) */ 
  std::vector<std::tuple<int, int, int, int, int, int, bool, bool>> problems = {
  
  
/*  m , n , k , lda , ldb , ldc , tA , tB */ 
std::make_tuple(192, 64, 784, 784, 784, 192, true, false), 
std::make_tuple(288, 64, 1440, 1440, 1440, 288, true, false), 
std::make_tuple(512, 192, 196, 196, 196, 512, true, false), 
std::make_tuple(576, 64, 2916, 2916, 2916, 576, true, false), 
std::make_tuple(576, 128, 360, 360, 360, 576, true, false), 
std::make_tuple(576, 128, 12544, 12544, 12544, 576, true, false), 
std::make_tuple(832, 256, 49, 49, 49, 832, true, false), 
std::make_tuple(1152, 128, 729, 729, 729, 1152, true, false), 
std::make_tuple(1152, 256, 196, 196, 196, 1152, true, false), 
std::make_tuple(1152, 256, 3136, 3136, 3136, 1152, true, false), 
std::make_tuple(1600, 32, 6308, 6308, 6308, 1600, true, false), 
std::make_tuple(2304, 512, 49, 49, 49, 2304, true, false), 
std::make_tuple(2304, 512, 784, 784, 784, 2304, true, false), 
std::make_tuple(4608, 512, 196, 196, 196, 4608, true, false), 
std::make_tuple(4608, 512, 49, 49, 49, 4608, true, false), 
std::make_tuple(4800, 32, 784, 784, 784, 4800, true, false), 
std::make_tuple(12800, 48, 196, 196, 196, 12800, true, false), 
std::make_tuple(20800, 128, 49, 49, 49, 20800, true, false), 

};
  
  bool isColMajor = true;
  bool tC = false;
  double alpha = 1.43235342345;
  double beta = 0.45348379373;
  char floattype = 'f'; 
  float allotted_time = 120.; 
  bool verbose = false;
  bool enforce_deterministic = false;
  
  /* We're just tracking the overall run time with these */
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms = end - start;
  float elapsed_seconds = fp_ms.count();
  
  
  for (unsigned prob_i = 0; prob_i < problems.size(); ++prob_i){
  
    auto problem = problems[prob_i];
    int m, n, k, lda, ldb, ldc;
    bool tA, tB;
    std::tie(m, n, k, lda, ldb, ldc, tA, tB) = problem;
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fp_ms = end - start;
    elapsed_seconds = fp_ms.count();
    
    std::cout << (prob_i + 1) <<  "/" <<  problems.size() << " \t m:" << m << " \t n:" << n << " \t k:" << k << " \t tA:" << tA << " \t tB:" << m << "  \t  elapsed time : " << elapsed_seconds << " [s]" << std::endl;    
    
    std::stringstream ss_logfile;
    ss_logfile << DIR_FOR_WRITING << "/backconvwrw/" << "at" << int(allotted_time) << "_m" << m  << "_n" << n  << "_k" << k  << "_tA" << tA  << "_tB" << tB << ".txt";   
    unsigned a_offset = 0;
    unsigned b_offset = 0;
    unsigned c_offset = 0;    
    unsigned n_postfind_runs = 0;
    basicfind<float>(isColMajor, tA, tB, tC, m, n, k, lda, ldb, ldc, a_offset, b_offset, c_offset, alpha, beta, floattype, allotted_time, verbose, ss_logfile.str(), enforce_deterministic, n_postfind_runs);    
  }
  
  return 0;
}



