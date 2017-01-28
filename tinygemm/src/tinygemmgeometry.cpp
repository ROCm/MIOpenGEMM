#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/consistencychecks.hpp>

namespace tinygemm{


TinyGemmGeometry::TinyGemmGeometry(bool isColMajor_, bool tA_, bool tB_, bool tC_, unsigned lda_, unsigned ldb_, unsigned ldc_, unsigned m_, unsigned n_, unsigned k_, unsigned a_offset_, unsigned b_offset_, unsigned c_offset_):isColMajor(isColMajor_), tA(tA_), tB(tB_), tC(tC_), lda(lda_), ldb(ldb_), ldc(ldc_), m(m_), n(n_), k(k_), a_offset(a_offset_), b_offset(b_offset_), c_offset(c_offset_) {
  consistencychecks::check_ldx_mnk_consistent(isColMajor,  tA,  tB,  tC,  lda,  ldb,  ldc,  m,  n,  k); //, a_offset, b_offset, c_offset
}

std::string TinyGemmGeometry::get_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << " tC:" << tC << " tA:" << tA << " tB:" << tB << " colMaj:" << isColMajor << " m:" << m << " n:" << n << " k:" << k << " lda:" << lda << " ldb:" << ldb << " ldc:" << ldc << " a_offset:" << a_offset << " b_offset:" << b_offset << " c_offset:" << c_offset;
  return geometry_stringstream.str();
}


std::string TinyGemmGeometry::get_networkconfig_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC" << tC << "_tA" << tA << "_tB" << tB << "_colMaj" << isColMajor << "_m" << m << "_n" << n << "_k" << k << "_lda" << lda << "_ldb" << ldb << "_ldc" << ldc;
  return geometry_stringstream.str();
}



float TinyGemmGeometry::get_distance(const TinyGemmGeometry & gg) const{
  /* problems which are "larger" are infinitely far away (as their tile might not fit) */
  
  float distance;
  
  if (tA != gg.tA || tB != gg.tB || isColMajor != gg.isColMajor || (m < std::min<unsigned>(600, gg.m)) || n < std::min<unsigned>(600, gg.n)){
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
