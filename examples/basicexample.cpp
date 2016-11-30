#include "basicfind.hpp"



template <typename TFloat>
void basicexample(){
  /* define the GEMM problem */
  bool isColMajor = false;
  bool tA = true;
  bool tB = false;
  bool tC = false;
  
  unsigned m = 100;
  unsigned n = 10;
  unsigned k = 5;

  unsigned lda = ( tA == isColMajor ? k : m ) + 0;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 0;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 1;
  
  unsigned a_offset = 0;
  unsigned b_offset = 0;
  unsigned c_offset = 0;
 
  /* These must be double, irrespective of the float type of the matrices */
  double alpha = 0.123;
  double beta = 0.456;//0.3;
  /* floattype should be 
   * 'f' for single-precision, 32-bit floats and 
   * 'd' for double-precision, 64-bit doubles */
  char floattype = (sizeof(TFloat) == 4) ? 'f' : 'd';
  /* define how long to search for, in seconds. No kernels will be compiled after this allotted time. */
  float allotted_time = 0.01;
  /* print output to terminal (true) or complete silence to terminal (false) */
  bool verbose = true;
  /* print output to logfile (non-empty string) or not (empty string) */
  /* MUST BE SET BY USER */
  std::string logfile("/home/james/tinygemm/examples/findlog.txt");
  /* enforce that the kernel is deterministic, or not. Note that 
   * for small problems, non-deterministic kernels are significantly (2x) faster */
  bool enforce_deterministic = false;
  unsigned n_postfind_runs = 5;//4;
  
  
  bool do_cpu_test = true;
  
  basicfind<TFloat>(isColMajor, tA, tB, tC, m, n, k, lda, ldb, ldc, a_offset, b_offset, c_offset, alpha, beta, floattype, allotted_time, verbose, logfile, enforce_deterministic, n_postfind_runs, do_cpu_test);
}

int main(){
  basicexample<float>(); /* or example<double> for dgemm example */
  return 0;
}
