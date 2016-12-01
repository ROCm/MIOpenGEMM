#include "basicfind.hpp"

template <typename TFloat>
void geometrytest(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k){

  unsigned lda = ( tA == isColMajor ? k : m ) + 0;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 0;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 0;
  
  unsigned a_offset = 0;
  unsigned b_offset = 0;
  unsigned c_offset = 0;
 
  /* These must be double, irrespective of the float type of the matrices */
  double alpha = 1.0; //0.1231321231243234523425;
  double beta = 1.0;//45343453456345344445346;
  char floattype = (sizeof(TFloat) == 4) ? 'f' : 'd';
  float allotted_time = 0.0001;
  bool verbose = true;
  std::string logfile(""); //home/james/tinygemm/examples/geometrytest.txt");
  bool enforce_deterministic = false;
  unsigned n_postfind_runs = 1;
  bool do_cpu_test = true;
    
  basicfind<TFloat>(isColMajor, tA, tB, tC, m, n, k, lda, ldb, ldc, a_offset, b_offset, c_offset, alpha, beta, floattype, allotted_time, verbose, logfile, enforce_deterministic, n_postfind_runs, do_cpu_test);
}

int main(){
  unsigned m = 100;
  
  
  //tC:0 tA:1 tB:0 colMaj:1 m:100 n:32 k:26939 lda:26939 ldb:26939 ldc:100 a_offset:0 b_offset:0 c_offset:0
  for (bool tC : {false}){
    for (bool isColMajor : {true}){
      for (bool tA : {true}){
	for (bool tB : {false}){
	  for  (unsigned n : {32}){//m - 29, m + 29}){
	    //TODO : fix tC = true case. 
	    geometrytest<float>(isColMajor, tA, tB, tC, m, n, 26939);
	  }  
	}
      }
    }
  }
  return 0;
}
