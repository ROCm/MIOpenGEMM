#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include "problemgeometry.hpp"
#include "consistencychecks.hpp"

namespace gemmgeometry{


Geometry::Geometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, unsigned a_offset, unsigned b_offset, unsigned c_offset):isColMajor(isColMajor), tA(tA), tB(tB), tC(tC), lda(lda), ldb(ldb), ldc(ldc), m(m), n(n), k(k), a_offset(a_offset), b_offset(b_offset), c_offset(c_offset) {
  consistencychecks::check_ldx_mnk_consistent(isColMajor,  tA,  tB,  tC,  lda,  ldb,  ldc,  m,  n,  k); //, a_offset, b_offset, c_offset
}

std::string Geometry::get_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << " tC:" << tC << " tA:" << tA << " tB:" << tB << " colMaj:" << isColMajor << " m:" << m << " n:" << n << " k:" << k << " lda:" << lda << " ldb:" << ldb << " ldc:" << ldc << " a_offset:" << a_offset << " b_offset:" << b_offset << " c_offset:" << c_offset;
  return geometry_stringstream.str();
}


float Geometry::get_distance(const Geometry & gg) const{
  //problems which are "larger" are infinitely far away (their tile might not fit...)
  
  float distance;
  
  if (tA != gg.tA || tB != gg.tB || isColMajor != gg.isColMajor || m < gg.m || n < gg.n){
    //std::cout << gg.get_string() << std::endl;
    //std::cout << get_string() << std::endl;
    distance = std::numeric_limits<float>::max();
  } 
   
  else{
    distance =  
    0.2*std::abs(float(k) - float(gg.k)) + 
    1.0*std::abs(float(m) - float(gg.m)) + 
    1.0*std::abs(float(n) - float(gg.n)) + 
    1.0*std::abs(float(lda) - float(gg.lda)) + 
    1.0*std::abs(float(ldb) - float(gg.ldb)) + 
    0.2*std::abs(float(ldc) - float(gg.ldc));
  }
  return distance;
}

}
