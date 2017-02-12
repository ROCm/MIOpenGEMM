#include <tinygemm/accuracytests.hpp>
#include <tinygemm/outputwriter.hpp>

namespace tinygemm {

namespace accuracytests {


template <typename TFloat>
void elementwise_compare(const TFloat * c_before, double beta, const TFloat * c_cpu, const TFloat * c_gpu, unsigned nels, tinygemm::outputwriting::OutputWriter & mowri){
  float max_relerr = 0;
  unsigned i_max = 0;
  for (unsigned i = 0; i < nels; ++i){
    float absdifference = std::abs(c_cpu[i] - c_gpu[i]);
    float sumabs = std::abs(c_cpu[i]) + std::abs(c_gpu[i]) + beta*std::abs(c_before[i]);
    float relerr = absdifference / std::max<float>(1e-9, sumabs);
    if (relerr > max_relerr){
      i_max = i;
      max_relerr = relerr;
    }
  }
  
    if (max_relerr > 0.01){
      std::stringstream ss;
      ss << "\nmax_relerr is above threshold, in basicfind.hpp. "; 
      ss << "\nIndex in c : " << i_max << "\nValue before gemm call : " << c_before[i_max] << "    .\nValue after call from gpu : "  << c_cpu[i_max] << ".  \nValue after call from cpu : " << c_gpu[i_max] << "  \nrelerr : " << max_relerr << std::endl;
      throw tinygemm::tinygemm_error(ss.str());
    }
  
  mowri << "max_relerr=" << max_relerr << Endl;
}

template void elementwise_compare(const float * c_before, double beta, const float * c_cpu, const float * c_gpu, unsigned nels, tinygemm::outputwriting::OutputWriter & mowri);

template void elementwise_compare(const double * c_before, double beta, const double * c_cpu, const double * c_gpu, unsigned nels, tinygemm::outputwriting::OutputWriter & mowri);


}
}
