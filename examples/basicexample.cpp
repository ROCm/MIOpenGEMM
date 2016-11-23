#include "basicfind.hpp"

template <typename TFloat>
void basicexample(){
  /* define the GEMM problem */
  bool isColMajor = true;
  bool tA = true;
  bool tB = false;
  bool tC = false;
  
  
  //tC:0 tA:1 tB:0 colMaj:1 m:2560 n:128 k:2560
  
  
  
  unsigned m = 1000;//6;    
  unsigned n = 1000;
  unsigned k = 1000;
  unsigned lda = (tA == isColMajor ? k : m ) + 3;
  unsigned ldb = (tB == isColMajor ? n : k ) + 5;
  unsigned ldc = (tC == isColMajor ? n : m ) + 11;
  /* These must be double, irrespective of the float type of the matrices */
  double alpha = 1.1;
  double beta = 0.3;
  /* floattype should be 
   * 'f' for single-precision, 32-bit floats and 
   * 'd' for double-precision, 64-bit doubles */
  char floattype = (sizeof(TFloat) == 4) ? 'f' : 'd';
  /* define how long to search for, in seconds. No kernels will be compiled after this allotted time. */
  float allotted_time = 3.0;
  /* print output to terminal (true) or complete silence to terminal (false) */
  bool verbose = true;
  /* print output to logfile (non-empty string) or not (empty string) */
  /* MUST BE SET BY USER */
  std::string logfile("/home/james/tinygemm/examples/findlog.txt");
  /* enforce that the kernel is deterministic, or not. Note that 
   * for small problems, non-deterministic kernels are significantly (2x) faster */
  bool enforce_deterministic = false;
  unsigned n_postfind_runs = 4;
  

  unsigned a_offset = 100;
  unsigned b_offset = 200;
  unsigned c_offset = 300;
  
  basicfind<TFloat>(isColMajor, tA, tB, tC, m, n, k, lda, ldb, ldc, a_offset, b_offset, c_offset, alpha, beta, floattype, allotted_time, verbose, logfile, enforce_deterministic, n_postfind_runs);
  
  
}

int main(){
  basicexample<float>(); /* or example<double> for dgemm example */
  return 0;
}


