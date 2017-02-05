#include "basicfind.hpp"

template <typename TFloat>
void geometrytest(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k){

  unsigned lda = ( tA == isColMajor ? k : m ) + 3;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 5;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 7;
  
  unsigned a_offset = 1;
  unsigned b_offset = 2;
  unsigned c_offset = 3;


   
  /* These must be double, irrespective of the float type of the matrices */
  float allotted_time = 0.001;
  /* set verbose to true if you want output to terminal */
  bool verbose = false;
  /* set logfile if you want output forked to file */
  std::string logfile("");
  bool enforce_deterministic = false;
  unsigned n_postfind_runs = 1;
  bool do_cpu_test = true;




  unsigned workspace_size = 3;
  unsigned workspace_offset = 4;      
  char floattype = sizeof(TFloat) == 4 ? 'f' : 'd';
  tinygemm::TinyGemmGeometry gg (isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
  tinygemm::TinyGemmOffsets offsets (a_offset, b_offset, c_offset, workspace_offset);    
  basicfind<TFloat>(gg, offsets, allotted_time, verbose, logfile, enforce_deterministic, n_postfind_runs, do_cpu_test);
}

int main(){
  unsigned m = 48;
  unsigned k = 80;
  unsigned testi = 0;
  for (bool tC : {false, true}){
    for (bool isColMajor : {false, true}){
      for (bool tA : {false, true}){
	for (bool tB : {false, true}){
	  for  (unsigned n : {m - 29, m + 30}){
	    testi += 1;
	    k += 1;
	    std::cout << "\ntest " << testi << "/32";
	    std::cout << "\nm=" << m << " n=" << n << " k=" << k << "\ntA=" << tA << " tB=" << tB << " tC=" << tC << " isColMajor=" << isColMajor << std::endl;
	    std::cout << "<float> ";
	    geometrytest<float>(isColMajor, tA, tB, tC, m, n, k);
	    std::cout << "<double> ";
	    geometrytest<double>(isColMajor, tA, tB, tC, m, n, k);
	  }
	}
      }
    }
  }
  return 0;
}
