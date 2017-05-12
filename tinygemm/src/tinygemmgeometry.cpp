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
unsigned TinyGemmGeometry::get_padless_dim(nsHP::eMat emat_x, bool isCoal) const{


  bool gate = (tX.at(emat_x) == isColMajor) == isCoal;

  if (emat_x == nsHP::matA){
    return gate ? k : m;
  }
  
  else if (emat_x == nsHP::matB){
    return gate ? n : k;
  }
  
  else if (emat_x == nsHP::matC){
    return gate ? n : m; 
  }
  
  else{
    throw tinygemm_error("unrecognised emat_x passed to get_coal in tinygemm geometry");
  }
}

unsigned TinyGemmGeometry::get_non_k_dim(nsHP::eMat emat_x) const{
  
  if (emat_x == nsHP::matA){
    return m;
  }
  
  else if (emat_x == nsHP::matB){
    return n;
  }
  
  else{
    throw tinygemm_error("invalid char passed to get_non_k_dim in tinygemm geometry, it should be either a or b");
  }  
}

  
unsigned TinyGemmGeometry::get_uncoal(nsHP::eMat emat_x) const{
  return get_padless_dim(emat_x, false);
}
 
unsigned TinyGemmGeometry::get_coal(nsHP::eMat emat_x) const{
  return get_padless_dim(emat_x, true);
}


bool TinyGemmGeometry::coal_is_pll_k(nsHP::eMat emat_x) const{
  /* proof : false, false, true should give 1 */
  return (static_cast<unsigned>(isColMajor) + static_cast<unsigned>(tX.at(emat_x)) + static_cast<unsigned>(emat_x == nsHP::matA)) % 2;
}
  

TinyGemmGeometry::TinyGemmGeometry(bool isColMajor_, bool tA_, bool tB_, bool tC_, unsigned lda_, unsigned ldb_, unsigned ldc_, unsigned m_, unsigned n_, unsigned k_, unsigned workspace_size_, char floattype_): isColMajor(isColMajor_), m(m_), n(n_), k(k_), workspace_size(workspace_size_), floattype(floattype_) {


  tX.resize(nsHP::nMats);
  tX[nsHP::matA] = tA_;
  tX[nsHP::matB] = tB_;  
  tX[nsHP::matC] = tC_;

  ldX.resize(nsHP::nMats);
  ldX[nsHP::matA] = lda_;
  ldX[nsHP::matB] = ldb_;  
  ldX[nsHP::matC] = ldc_;
  
  
  if (floattype != 'd' and floattype != 'f'){
    throw tinygemm::tinygemm_error("floattype should be one of 'f' and 'd' (in TinyGemmGeometry constructor)");
  }
    
  consistencychecks::check_ldx_mnk_consistent(isColMajor, tX[nsHP::matA], tX[nsHP::matB], tX[nsHP::matC],  ldX[nsHP::matA],  ldX[nsHP::matB],  ldX[nsHP::matC],  m,  n,  k);
  
  derived.reset(floattype); 

}

std::string TinyGemmGeometry::get_string() const{
  
  return get_networkconfig_string();

}

std::string TinyGemmGeometry::get_networkconfig_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC" << tX[nsHP::matC]  << "_tA" << tX[nsHP::matA] << "_tB" << tX[nsHP::matB] << "_colMaj" << isColMajor << "_m" << m << "_n" << n << "_k" << k << "_lda" << ldX[nsHP::matA] << "_ldb" << ldX[nsHP::matB] << "_ldc" << ldX[nsHP::matC] << "_ws" << workspace_size << "_f" << derived.float_size_bits;
  return geometry_stringstream.str();
}





}
