#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/consistencychecks.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{

TinyGemmOffsets::TinyGemmOffsets(unsigned oa_, unsigned ob_, unsigned oc_, unsigned oworkspace_, unsigned tail_off_a_, unsigned tail_off_b_, unsigned tail_off_c_):oa(oa_), ob(ob_), oc(oc_), oworkspace(oworkspace_), tail_off_a(tail_off_a_), tail_off_b(tail_off_b_), tail_off_c(tail_off_c_) {}

const unsigned & TinyGemmOffsets::operator[](char x) const{
  if (x == 'a'){
    return oa;
  }
  else if (x == 'b'){
    return ob;
  }
  else if (x == 'c'){
    return oc;
  }
  else if (x == 'w'){
    return oworkspace;
  }
  
  else{
    throw tinygemm_error(std::string("unrecognised char passed to operator[](char x) of TinyGemmOffsets. Should be one of a,b,c,w, not ") + x);
  }
}

 
 
void TinyGemmGeometryDerived::reset(char floattype){
  if (floattype == 'f'){
    float_size_bytes = sizeof(float);
  }
  else if (floattype == 'd'){
    float_size_bytes = sizeof(double);
  }
  else{
    throw tinygemm_error("what is this floattype : " + floattype + std::string(" ?? in tinygemmgeometry"));
  }
  float_size_bits = 8*float_size_bytes;
}


/* return one of the dimensions of matrix a,b,c. this has nothing to do with lda, ldb, ldc. 
 * isCoal : the coalesced dimesion? For example, for 'a' which is is m x k, 
 * if tC = false, isColMajor = false, isCoal = true, then k is returned as k is the coalesced dim. 
 * (false == false) == true  evaluates to true, so gate is true, so m is returned */
unsigned TinyGemmGeometry::get_padless_dim(char x, bool isCoal) const{

  bool gate = (tC == isColMajor) == isCoal;

  if (x == 'a'){
    return gate ? k : m;
  }
  
  else if (x == 'b'){
    return gate ? n : k;
  }
  
  else if (x == 'c'){
    return gate ? n : m; 
  }
  
  else{
    throw tinygemm_error("unrecognised char passed to get_coal in tinygemm geometry");
  }
}

unsigned TinyGemmGeometry::get_non_k_dim(char x) const{
  
  if (x == 'a'){
    return m;
  }
  
  else if (x == 'b'){
    return n;
  }
  
  else{
    throw tinygemm_error("invalid char passed to get_non_k_dim in tinygemm geometry, it should be either a or b");
  }  
}

unsigned TinyGemmGeometry::get_ld(char x) const {
  
  if (x == 'a'){
    return lda;
  }
  
  else if (x == 'b'){
    return ldb;
  }
  
  else if (x == 'c'){
    return ldc;
  }
  
  else{
    throw tinygemm_error("unrecognised char passed to get_ld in tinygemm geometry");
  }
  
}
  
unsigned TinyGemmGeometry::get_uncoal(char x) const{
  return get_padless_dim(x, false);
}
 
unsigned TinyGemmGeometry::get_coal(char x) const{
  return get_padless_dim(x, true);
} 


bool TinyGemmGeometry::coal_is_pll_k(char x) const{
  if ((x != 'a') && (x != 'b')){
    throw tinygemm_error("parameter to coal_is_pll_k should be 'a' or 'b'");
  }
  
  /* proof : false, false, true should give 1 */
  return (static_cast<unsigned>(isColMajor) + static_cast<unsigned>(get_tX(x)) + static_cast<unsigned>(x == 'a')) % 2;
}
  

TinyGemmGeometry::TinyGemmGeometry(bool isColMajor_, bool tA_, bool tB_, bool tC_, unsigned lda_, unsigned ldb_, unsigned ldc_, unsigned m_, unsigned n_, unsigned k_, unsigned workspace_size_, char floattype_): isColMajor(isColMajor_), tA(tA_), tB(tB_), tC(tC_), lda(lda_), ldb(ldb_), ldc(ldc_), m(m_), n(n_), k(k_), workspace_size(workspace_size_), floattype(floattype_) {

  
  if (floattype != 'd' and floattype != 'f'){
    throw tinygemm::tinygemm_error("floattype should be one of 'f' and 'd' (in TinyGemmGeometry constructor)");
  }
    
  consistencychecks::check_ldx_mnk_consistent(isColMajor,  tA,  tB,  tC,  lda,  ldb,  ldc,  m,  n,  k); //, a_offset, b_offset, c_offset
  
  derived.reset(floattype); //tC, isColMajor, n, m, 

}

std::string TinyGemmGeometry::get_string() const{
  
  return get_networkconfig_string();
  //std::stringstream geometry_stringstream;
  //geometry_stringstream << " tC:" << tC << " tA:" << tA << " tB:" << tB << " colMaj:" << isColMajor << " m:" << m << " n:" << n << " k:" << k << " lda:" << lda << " ldb:" << ldb << " ldc:" << ldc  << " workspace_size:" << workspace_size << " floattype:" << floattype;
  //return geometry_stringstream.str();
}

std::string TinyGemmGeometry::get_networkconfig_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC" << tC << "_tA" << tA << "_tB" << tB << "_colMaj" << isColMajor << "_m" << m << "_n" << n << "_k" << k << "_lda" << lda << "_ldb" << ldb << "_ldc" << ldc << "_ws" << workspace_size << "_f" << derived.float_size_bits;
  return geometry_stringstream.str();
}


bool TinyGemmGeometry::get_tX(char x) const{
  if (x == 'a' || x == 'A'){
    return tA;
  }
  else if (x == 'b' || x == 'B'){
    return tB;
  }
  
  else if (x == 'c' || x == 'C'){
    return tC;
  }
  
  else{
    throw tinygemm_error("char x unrecognised in get_tX in tiny gemm geom");
  }
}


float TinyGemmGeometry::get_distance(const TinyGemmGeometry & gg) const{
  /* problems which are "larger" are infinitely far away (as their tile might not fit) */
  
  
  
  float distance;
  
  if (workspace_size < gg.workspace_size || floattype != gg.floattype || tA != gg.tA || tB != gg.tB || isColMajor != gg.isColMajor || (m < std::min<unsigned>(600, gg.m)) || n < std::min<unsigned>(600, gg.n)){
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
