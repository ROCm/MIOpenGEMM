#ifndef SLOWCPUGEMM
#define SLOWCPUGEMM

#include <vector>
#include <string>

#include "outputwriter.hpp"

namespace slowcpugemm{



template <typename TFloat> // available for float and double (instantiations in slowcpugemm.cpp)
void gemm_3fors_cpu(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, const TFloat * a, const TFloat * b, TFloat * c, TFloat alpha, TFloat beta, std::vector<std::string> algs, outputwriting::OutputWriter & mowri);
  

} //namespace



#endif
