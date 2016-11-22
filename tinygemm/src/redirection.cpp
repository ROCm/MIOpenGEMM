#include <algorithm>
#include <stdexcept>
#include  <CL/cl.h> 

#include "redirection.hpp"

namespace redirection{
  
template <typename T>
void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, T & a, T  & b){

  throw std::runtime_error("In tinygemm :: redirection.cpp. Redirection has been deprecated, it needs fixing up since the introduction of a_offset, b_offset.");
  
  if (isColMajor == false){
    
    /* minimal changes to get into row major : */
    std::swap(tA, tB);
    std::swap(a, b);
    std::swap(lda, ldb);
    std::swap(m,n);
    isColMajor = true;
    
    /* it might still be TT or m < n && tA + tB == 1, redirect again */
    redirect<T>(isColMajor, tA, tB, tC, m, n, lda, ldb, a, b);//isColMajor, tA, tB, tC, a, b, c, lda, ldb, ldc, m, n, k);
  }
  
  else if (tA == true && tB == true){

    tC = ~tC;
    tA = false;
    tB = false;
    std::swap(lda, ldb);
    std::swap(a,b);
    std::swap(m,n);
  }
  
  else if (m  > n && ((tA == true  && tB == false) || (tA == false  && tB == true))) {

    tC = ~tC;
    std::swap(lda, ldb);
    std::swap(a,b);
    std::swap(m,n);
  }
}


template void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, const double * & a, const double * & b);

template void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, const float * & a, const float * & b);

template void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, cl_mem & a, cl_mem & b);


void confirm_redirection(bool isColMajor, bool tA, bool tB, unsigned m, unsigned n){
  if (isColMajor == false) {
    throw std::runtime_error("isColMajor == false : see symmetry_red document for redirection and implement");
  }
  
  else{
    if (tA == true && tB == true){
      throw std::runtime_error("both matrices transposed : see symmetry_red document for redirection and implement");
    }
   
    else if ((tA == true && tB == false) || (tA == false && tB == true)){
      if (m > n){
        throw std::runtime_error("tA + tB = 1 with m > n : see symmetry_red document for redirection to m <= n");
      }
    }
  }
}

}
  
//TODO : add redirect for gpu opencl buffers.
