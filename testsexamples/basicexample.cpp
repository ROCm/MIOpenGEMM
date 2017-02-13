#include "basicfind.hpp"



template <typename TFloat>
void basicexample(){
  /* define the GEMM problem */
  bool isColMajor = true;
  bool tA = false;
  bool tB = false;
  bool tC = false;
  
  unsigned m = 322;
  unsigned n = 400;
  unsigned k = 500;

  unsigned lda = ( tA == isColMajor ? k : m ) + 1;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 2;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 3;
  
  unsigned a_offset = 1;
  unsigned b_offset = 2;
  unsigned c_offset = 3;


  unsigned tail_off_a = 1000;
  unsigned tail_off_b = 1000;
  unsigned tail_off_c = 1000;
 
  /* These must be double, irrespective of the float type of the matrices */
  //double alpha = 0.123;
  //double beta = 0.456;//0.3;
  /* floattype should be 
   * 'f' for single-precision, 32-bit floats and 
   * 'd' for double-precision, 64-bit doubles */
  
  /* define how long to search for, in seconds. No kernels will be compiled after this allotted time. */
  float allotted_time = 1.01;
  /* print output to terminal (true) or complete silence to terminal (false) */
  bool verbose = true;
  /* print output to logfile (non-empty string) or not (empty string) */
  /* MUST BE SET BY USER */
  std::string logfile("basicexample-findlog.txt");
  /* enforce that the kernel is deterministic, or not. Note that 
   * for small problems, non-deterministic kernels are significantly (2x) faster */
  bool enforce_deterministic = false;
  unsigned n_postfind_runs = 5; //4;
  bool do_cpu_test = true;  
  //basicfind<TFloat>(isColMajor, tA, tB, tC, m, n, k, lda, ldb, ldc, a_offset, b_offset, c_offset, alpha, beta, allotted_time, verbose, logfile, enforce_deterministic, n_postfind_runs, do_cpu_test);
  


  unsigned workspace_size = 3;
  unsigned workspace_offset = 4;      
  char floattype = 'f';
  tinygemm::TinyGemmGeometry gg (isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
  tinygemm::TinyGemmOffsets offsets (a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c);    
  basicfind<float>(gg, offsets, allotted_time, verbose, logfile, enforce_deterministic, n_postfind_runs, do_cpu_test);    
  
  
    
}

int main(){
  basicexample<float>(); /* or example<double> for dgemm example */
  return 0;
}
