#include <chrono>
#include <stdexcept>

#include "slowcpugemm.hpp"
#include "outputwriter.hpp"
#include "redirection.hpp"


namespace slowcpugemm{

template<typename TFloat>
class NNInner{
  public:
  inline TFloat operator() (const TFloat * a, const TFloat * b, unsigned x, unsigned y, unsigned lda, unsigned ldb, unsigned k){
    TFloat inner = 0;
    for (unsigned  z = 0; z < k; ++z){
      inner += a[x + z*lda]*b[y*ldb + z];
    }
    return inner;
  }
};


template<typename TFloat>
class NTInner{
  public:
  inline TFloat operator() (const TFloat * a, const TFloat * b, unsigned x, unsigned y, unsigned lda, unsigned ldb, unsigned k){
    TFloat inner = 0;
    for (unsigned  z = 0; z < k; ++z){
      inner += a[x + z*lda]*b[y + z*ldb];
    }
    return inner;
  }
};

template<typename TFloat>
class TNInner{
  public:
  inline TFloat operator() (const TFloat * a, const TFloat * b, unsigned x, unsigned y, unsigned lda, unsigned ldb, unsigned k){
    TFloat inner = 0;
    for (unsigned  z = 0; z < k; ++z){
      inner += a[x*lda + z]*b[y*ldb + z];
    }
    return inner;
  }
};



template <typename TFloat, class FInner>
void gemm_3fors_generic_cpu(unsigned m, unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldc, const TFloat * a, const TFloat * b, TFloat * c, TFloat alpha, TFloat beta, bool tC){
  /* at this point, must be column contiguous (ala fortran)
 * this is a generic slow matrix multiplier for NN, TN, NT. 
 * NN, TN, NT will have different FInner template parameters
 * TT should have have been redirected to NN at this point*/
   
  FInner finner;
  
  /* For rows of C */
  for (unsigned  x = 0; x < m; ++x){
    /* For columns of C */
    for (unsigned  y = 0; y < n; ++y){
      /* Set the index of the element in C we're setting, */
      unsigned target_index;
      if (tC == false){
        target_index = x + y*ldc;
      }
      else{
        target_index = y + x*ldc;
      }
      /* and set it */
      c[target_index] *= beta;
      c[target_index] += alpha*finner(a, b, x, y, lda, ldb, k);
    }
  }
}

template <typename TFloat>
void gemm_3fors_cpu(bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldc, const TFloat * a, const TFloat * b, TFloat * c, TFloat alpha, TFloat beta){
    
  if (tA == true && tB == true){
    throw std::runtime_error("tA and tB should have been redirected before calling gemm_3fors_cpu");
  }
    
  else if (tA == false && tB == false){
    gemm_3fors_generic_cpu<TFloat, NNInner<TFloat>>(m, n, k, lda, ldb, ldc, a, b, c, alpha, beta, tC); //tC, m, n, k, alpha, a, lda, b, ldb,  beta, c, ldc);
  }
  
  
  else{
    if (m > n){
      throw std::logic_error("m > n should have been redirected before calling gemm_3fors_cpu");
    }
  
    if (tA == false && tB == true){
      gemm_3fors_generic_cpu<TFloat, NTInner<TFloat>>(m, n, k, lda, ldb, ldc, a, b, c, alpha, beta, tC); //tC, m, n, k, alpha, a, lda, b, ldb,  beta, c, ldc);
    }
    else if (tA == true && tB == false){
      gemm_3fors_generic_cpu<TFloat, TNInner<TFloat>>(m, n, k, lda, ldb, ldc, a, b, c, alpha, beta, tC); //tC, m, n, k, alpha, a, lda, b, ldb,  beta, c, ldc);
    }
    
    else{
      throw std::runtime_error("this will never happen");
    }
  }
}



void check_cpu_algs(std::vector<std::string> cpu_algs){

  std::vector<std::string> known_algs = {"3fors"};
  
  for (auto & alg : cpu_algs){
    bool isknown = false;
    for (auto & kalg : known_algs){ 
      if (alg.compare(kalg) == 0){
        isknown = true;
        break;
      }
    }
    
    if (isknown == false){
      std::string errm = "unrecognised cpu algorithm, ";
      errm += alg;
      errm += "\n";
      throw std::runtime_error(errm);
    }
  }
}

  template <typename TFloat>
  void gemm_3fors_cpu(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, const TFloat * a, const TFloat * b, TFloat * c, TFloat alpha, TFloat beta, std::vector<std::string> algs, outputwriting::OutputWriter & mowri){
    
    check_cpu_algs(algs);
    
    //redirect.
    redirection::redirect<const TFloat * >(isColMajor, tA, tB, tC, m, n, lda, ldb, a, b);
    redirection::confirm_redirection(isColMajor, tA, tB, m, n);
    
    for (auto & alg : algs){
      mowri << "launching cpu algorithm : " << alg << Endl;      
      auto t0 = std::chrono::high_resolution_clock::now();
      if (alg.compare("3fors") == 0){
        gemm_3fors_cpu(tA, tB, tC, m, n, k, lda, ldb, ldc, a, b, c, alpha, beta);
      }
      
      
      auto t1 = std::chrono::high_resolution_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      mowri << "elapsed time : " << elapsed_time * 1e-6 << " [s] " << Endl;
    }
  }
  
  template void gemm_3fors_cpu(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, const float * a, const float * b, float * c, float alpha, float beta, std::vector<std::string> algs, outputwriting::OutputWriter & mowri);

  template void gemm_3fors_cpu(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, const double * a, const double * b, double * c, double alpha, double beta, std::vector<std::string> algs, outputwriting::OutputWriter & mowri);  
  
} //namespace
