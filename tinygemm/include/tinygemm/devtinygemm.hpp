#ifndef DEGEMMAPIQQ_HPP
#define DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <vector>
#include <string>
namespace tinygemm{
namespace dev{


template <typename TFloat>
void benchgemm(const std::vector<hyperparams::HyperParams> & hps,         
unsigned n_runs, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const TFloat * a, const TFloat * b, const TFloat * c, bool verbose = true, std::string logfile = "");

template <typename TFloat>
void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const TFloat * a, const TFloat * b,
const TFloat * c, const TFloat * c_true_for_test = nullptr, bool verbose = true, std::string logfile = "");



//void hello();

//template <typename TFloat>		// available for float and double (instantiations in devgemmapi.cpp)
//void benchgemm (bool isColMajor,	// false for row contiguous (ala C) and true for column contiguous (ala Fortran) 
//bool tA,		
//bool tB, 
//bool tC,
//unsigned m, 
//unsigned n, 
//unsigned k, 
//TFloat alpha, 
//const TFloat * a, 
//unsigned lda,
//unsigned a_offset,
//const TFloat * b, 
//unsigned ldb, 
//unsigned b_offset,
//TFloat beta, 
//TFloat * c, 
//unsigned ldc,
//unsigned c_offset,
//std::vector<std::string> cpu_algs,															// which cpu algorithms to run 
//std::vector<tinygemm::hyperparams::HyperParams > hps,
//bool capture_output, 
//std::string & output,
//const TFloat * c_true_for_test, 																// TODO : comment on format of c_true_for_test. 
//unsigned do_test, 
//unsigned n_runs, 
//std::string outputfilename, 
//bool findfirst, 
//float allotted_time,
//bool enforce_deterministic);


}
}

#endif
