#include <algorithm>
#include "tinygemmerror.hpp"
#include  <CL/cl.h> 

#include "redirection.hpp"

#include <iostream>
namespace tinygemm{
namespace redirection{
  
template <typename T>
void redirect_base(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, T & a, T  & b){  
  if (isColMajor == false){
    /* minimal changes to get into row major : */
    std::swap(tA, tB);
    std::swap(a, b);
    std::swap(m, n);
    isColMajor = true;
    /* it might still be TT or m < n && tA + tB == 1, redirect again */
    redirect_base<T>(isColMajor, tA, tB, tC, m, n, a, b);
  }
  
  else if (tA == true && tB == true){
    tC = tC == true ? false : true;
    tA = false;
    tB = false;
    std::swap(a,b);
    std::swap(m,n);
  }
  
  else if (m  > n && ((tA == true  && tB == false) || (tA == false  && tB == true))) {
    tC = tC == true ? false : true;
    std::swap(a,b);
    std::swap(m,n);
  }
}

template <typename TFloat> 
class MatrixBundle{
  public : 
    const TFloat * x;
    unsigned ldx;
    unsigned x_offset;
    MatrixBundle(const TFloat * x, unsigned ldx, unsigned x_offset):x(x), ldx(ldx), x_offset(x_offset) {}
};

template <typename TFloat>
void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, unsigned & a_offset, unsigned & b_offset, const TFloat * & a, const TFloat * & b){

  MatrixBundle<TFloat> a_bundle (a, lda, a_offset);
  MatrixBundle<TFloat> b_bundle (b, ldb, b_offset);
  redirect_base<MatrixBundle<TFloat>>(isColMajor, tA, tB, tC, m, n, a_bundle, b_bundle);
  
  a = a_bundle.x;
  lda = a_bundle.ldx;
  a_offset = a_bundle.x_offset;
  
  b = b_bundle.x;
  ldb = b_bundle.ldx;
  b_offset = b_bundle.x_offset;  

}


template void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, unsigned & a_offset, unsigned & b_offset, const double *  & a, const double * & b);
template void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, unsigned & a_offset, unsigned & b_offset, const float *  & a, const float * & b);

void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, std::string & a, std::string & b){
  redirect_base<std::string>(isColMajor, tA, tB, tC, m, n, a, b);
}

void confirm_redirection(bool isColMajor, bool tA, bool tB, unsigned m, unsigned n){
  if (isColMajor == false) {
    throw tinygemm_error("isColMajor == false : see symmetry_red document for redirection and implement");
  }
  
  else{
    if (tA == true && tB == true){
      throw tinygemm_error("both matrices transposed : see symmetry_red document for redirection and implement");
    }
   
    else if ((tA == true && tB == false) || (tA == false && tB == true)){
      if (m > n){
        throw tinygemm_error("tA + tB = 1 with m > n : see symmetry_red document for redirection to m <= n");
      }
    }
  }
}

}
}
  
//TODO : add redirect for gpu opencl buffers.
