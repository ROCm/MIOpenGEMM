#include <MIOpenGEMM/accuracytests.hpp>
#include <MIOpenGEMM/outputwriter.hpp>
#include <algorithm>

namespace MIOpenGEMM {

namespace accuracytests {


template <typename TFloat>
void elementwise_compare(const TFloat * c_before, double beta, const TFloat * c_cpu, const TFloat * c_gpu, unsigned nels, outputwriting::OutputWriter & mowri){
  
  
  // 1 in 100.
  double threshold = 0.01;
  double max_relerr = 0.;
  unsigned i_max = 0;
  
  std::vector<unsigned> violating_indices = {};
  std::vector<double> violating_margins = {};
 
  
  for (unsigned i = 0; i < nels; ++i){
    
    double absdifference = std::abs(c_cpu[i] - c_gpu[i]);
    
    double sumabs = 0.3333*(std::abs(c_cpu[i]) + std::abs(c_gpu[i]) + beta*std::abs(c_before[i]));
    double relerr = absdifference / std::max<double>(1e-9, sumabs);
    
    
    if (relerr > threshold){
      violating_indices.push_back(i);
      violating_margins.push_back(relerr);
    }
    
    if (relerr > max_relerr){
      i_max = i;
      max_relerr = relerr;
    }
  }
  
    if (max_relerr > threshold){
      std::stringstream ss;
      ss << "\nmax_relerr is above threshold, in basicfind.hpp. "; 
      ss << "\nIndex in c : " << i_max << "\nValue before gemm call : " << c_before[i_max] << ". \nValue after call from cpu : "  << c_cpu[i_max] << ".  \nValue after call from gpu : " << c_gpu[i_max] << "  \nrelerr : " << max_relerr << "\n";
      ss << "the first violating indices (above the threshold of " << threshold << ") were: \n";
      for (unsigned bl= 0; bl < std::min<size_t>(10, violating_indices.size()); ++bl){
        ss << " " << violating_indices[bl] << " (" << violating_margins[bl] << ") ";
      }
      
      
      ss << "\n{c before}  (cpu)  [gpu]\n";
      for (unsigned i = 0; i < std::min<unsigned>(nels, 16); ++i){
        ss << "{" << c_before[i] << "}  (" <<  c_cpu[i]  << ")  ["  << c_gpu[i] << "]" << "\n"; 
       } 
      
      throw miog_error(ss.str());
    }
  
  mowri << "max_relerr=" << max_relerr << Endl;
}

template void elementwise_compare(const float * c_before, double beta, const float * c_cpu, const float * c_gpu, unsigned nels, outputwriting::OutputWriter & mowri);

template void elementwise_compare(const double * c_before, double beta, const double * c_cpu, const double * c_gpu, unsigned nels, outputwriting::OutputWriter & mowri);


}
}
