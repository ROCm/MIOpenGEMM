#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include "problemgeometry.hpp"
#include "consistencychecks.hpp"

namespace gemmgeometry{


Geometry::Geometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k):isColMajor(isColMajor), tA(tA), tB(tB), tC(tC), lda(lda), ldb(ldb), ldc(ldc), m(m), n(n), k(k) {
  consistencychecks::check_ldx_mnk_consistent(isColMajor,  tA,  tB,  tC,  lda,  ldb,  ldc,  m,  n,  k);
}

std::string Geometry::get_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << " tC:" << tC << " tA:" << tA << " tB:" << tB << " colMaj:" << isColMajor << " m:" << m << " n:" << n << " k:" << k << " lda:" << lda << " ldb:" << ldb << " ldc:" << ldc;
  return geometry_stringstream.str();
}


float Geometry::distance(const Geometry & gg){
  //problems which are "larger" are infinitely far away (their tile might not fit...)
  
  float distance;
  
  if (tA != gg.tA || tB != gg.tB || m < gg.m || n < gg.n){
    distance = std::numeric_limits<float>::max();
  } 
   
  else{
    distance =  std::abs(float(k) - float(gg.k)) + std::abs(float(m) - float(gg.m)) + std::abs(float(n) - float(gg.n)) + std::abs(float(lda) - float(gg.lda)) + std::abs(float(ldb) - float(gg.ldb)) + std::abs(float(ldc) - float(gg.ldc));
  }
  return distance;
}

}
