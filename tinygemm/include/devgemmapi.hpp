#ifndef DEGEMMAPIQQ_HPP
#define DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <vector>
#include <string>
namespace devgemmns{

void hello();

template <typename TFloat>		// available for float and double (instantiations in devgemmapi.cpp)
void benchgemm (bool isColMajor,	// false for row contiguous (ala C) and true for column contiguous (ala Fortran) 
bool tA,		
bool tB, 
bool tC,
unsigned m, 
unsigned n, 
unsigned k, 
TFloat alpha, 
const TFloat * a, 
unsigned lda, 
const TFloat * b, 
unsigned ldb, 
TFloat beta, 
TFloat * c, 
unsigned ldc,
std::vector<std::string> cpu_algs,															// which cpu algorithms to run 
std::vector<std::vector<std::string> > gpu_kernel_filenames,		// see comment below
bool capture_output, 
std::string & output,
const TFloat * c_true_for_test, 																// TODO : comment on format of c_true_for_test. 
unsigned do_test, 
unsigned n_runs, 
std::string outputfilename, 
bool findfirst, 
float allotted_time,
bool enforce_deterministic);

/* gpu_kernel_filenames needs some explaining.
* It is a vector containing filenames of opencl kernels to run. 
* Example : 
* [ ["kern0_0.cl"],  
*   ["kern1_0.cl", kern1_2.cl",  kern1_2.cl"]  ]
* Currently (2/11/2016) only one kernel per list is supported. 
*/




} //namespace devgemmns

#endif
