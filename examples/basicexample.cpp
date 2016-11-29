#include "basicfind.hpp"


template <typename TFloat>
void basicexample(){
  /* define the GEMM problem */
  bool isColMajor = true;


  bool tC = false;
  bool tA = true;
  bool tB = false;
  
  
  //20800, 128, 49
   
  //m ,  n ,  k , lda ,ldb, ldc, tA, tB
  
  //1, 0, 27, 64, 50176, 50176, 50176, 64 
  
  unsigned m = 20800;//100;//1600; //3000;    
  unsigned n = 128;//400;
  unsigned k = 49;//26939;//500;

  unsigned lda = (tA == isColMajor ? k : m ) + 0;//3;
  unsigned ldb = (tB == isColMajor ? n : k ) + 0;//5;
  unsigned ldc = (tC == isColMajor ? n : m ) + 0;//11;

  unsigned a_offset = 0;//100;
  unsigned b_offset = 0;//200;
  unsigned c_offset = 0;//300;

  /* These must be double, irrespective of the float type of the matrices */
  double alpha = 1.0;
  double beta = .2;//0.3;
  /* floattype should be 
   * 'f' for single-precision, 32-bit floats and 
   * 'd' for double-precision, 64-bit doubles */
  char floattype = (sizeof(TFloat) == 4) ? 'f' : 'd';
  /* define how long to search for, in seconds. No kernels will be compiled after this allotted time. */
  float allotted_time = 60.0;
  /* print output to terminal (true) or complete silence to terminal (false) */
  bool verbose = true;
  /* print output to logfile (non-empty string) or not (empty string) */
  /* MUST BE SET BY USER */
  std::string logfile("/home/james/tinygemm/examples/findlog.txt");
  /* enforce that the kernel is deterministic, or not. Note that 
   * for small problems, non-deterministic kernels are significantly (2x) faster */
  bool enforce_deterministic = false;
  unsigned n_postfind_runs = 5;//4;
  
  
  basicfind<TFloat>(isColMajor, tA, tB, tC, m, n, k, lda, ldb, ldc, a_offset, b_offset, c_offset, alpha, beta, floattype, allotted_time, verbose, logfile, enforce_deterministic, n_postfind_runs);
  
  
}

int main(){
  basicexample<float>(); /* or example<double> for dgemm example */
  return 0;
}


