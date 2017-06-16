#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <map>
#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/stringutilbase.hpp>

namespace MIOpenGEMM{


std::vector<char> get_matChars(){
  std::vector<char> v1;
  v1.resize(nsHP::nMats);
  v1[nsHP::matA] = 'a';
  v1[nsHP::matB] = 'b';
  v1[nsHP::matC] = 'c'; 
  return v1;
}

std::vector<char> matChars = get_matChars();

Offsets::Offsets(unsigned oa_, unsigned ob_, unsigned oc_, unsigned oworkspace_, unsigned tail_off_a_, unsigned tail_off_b_, unsigned tail_off_c_):oa(oa_), ob(ob_), oc(oc_), oworkspace(oworkspace_), tail_off_a(tail_off_a_), tail_off_b(tail_off_b_), tail_off_c(tail_off_c_) {}

const unsigned & Offsets::operator[](char x) const{
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
    throw  miog_error(std::string("unrecognised char passed to operator[](char x) of Offsets. Should be one of a,b,c,w, not ") + x);
  }
}

 

char get_floattype(unsigned nbits){

  char ft = 'x';
  if (nbits == 8*sizeof(float)){
    ft = 'f';
  }
  else if (nbits == 8*sizeof(double)){
    ft = 'd';
  }
  else{
    throw  miog_error("what is the floattype with number of bints : " + std::to_string(nbits) + std::string(" ?? in get_floattype of geometry"));
  }
  return ft;
}

void GeometryDerived::reset(char floattype){
  if (floattype == 'f'){
    float_size_bytes = sizeof(float);
  }
  else if (floattype == 'd'){
    float_size_bytes = sizeof(double);
  }
  else{
    throw  miog_error("what is this floattype : " + floattype + std::string(" ?? in reset of geometry"));
  }
  float_size_bits = 8*float_size_bytes;
}


/* return one of the dimensions of matrix a,b,c. this has nothing to do with lda, ldb, ldc. 
 * isCoal : the coalesced dimesion? For example, for 'a' which is m x k, 
 * if tA = false, isColMajor = false, isCoal = true, then k is returned as k is the coalesced dim. 
 * (false == false) == true  evaluates to true, so gate is true, so m is returned */
unsigned Geometry::get_padless_dim(nsHP::eMat emat_x, bool isCoal) const{


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
    throw  miog_error("unrecognised emat_x passed to get_coal in get_padless_dim of geometry");
  }
}

unsigned Geometry::get_non_k_dim(nsHP::eMat emat_x) const{
  
  if (emat_x == nsHP::matA){
    return m;
  }
  
  else if (emat_x == nsHP::matB){
    return n;
  }

  else{
    throw  miog_error("invalid char passed to get_non_k_dim in get_non_k_dim of geometry, it should be either a or b");
  }  
}

void Geometry::check_ldx_consistent() const{

  bool error = false;
  for (auto x : {nsHP::matA, nsHP::matB, nsHP::matC}){
    if (ldX[x] < get_coal(x)){
      error = true;
    }
  }
  
  if (error == true){

    std::stringstream errm_ss;
    


    errm_ss << "Checking that lda, ldb, ldc are consistent with m,n,k. ";
    errm_ss << "In particulary, checking that ldx is at least as large as coalesced dimension, coal_x (for x in {a,b,c}) given by:  ";
    errm_ss << "coal_a = (tA == isColMajor ? k : m),  ";
    errm_ss << "coal_b = (tB == isColMajor ? n : k),  ";
    errm_ss << "coal_c = (tC == isColMajor ? k : m).  ";
    errm_ss << "\n\n";

    errm_ss << "ldx = coal_x + pad_x, and so for consisteny it must be true that ldx >= coal_x (can't have negative pad_x).  ";
    errm_ss << "As an example, if tA = false and isColMajor = false, then coal_a = k.  ";
    errm_ss << "A full table of the lower bounds of ldx for x in {a,b,c} can be found at, https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm.  ";
    errm_ss << "\n\n";
    
    errm_ss << "The particular geometry received by in geometry check_ldx_consistent is  ";
    errm_ss << get_string();
    errm_ss << ", and the problems detected are:  ";
    
    
    std::vector<char> m_chars (nsHP::nMats);
    m_chars[nsHP::matA] = 'a';
    m_chars[nsHP::matB] = 'b';
    m_chars[nsHP::matC] = 'c';
    
  
    
    for (auto x : {nsHP::matA, nsHP::matB, nsHP::matC}){
      if (ldX[x] < get_coal(x)){
        errm_ss << "ld" << m_chars[x] << " (" << ldX[x] << ") <  coal_" << m_chars[x] << " (" << get_coal(x) << ").  ";
      }
    }
        
    throw  miog_error(errm_ss.str());
  }
  
  
  
}

  
unsigned Geometry::get_uncoal(nsHP::eMat emat_x) const{
  return get_padless_dim(emat_x, false);
}
 
unsigned Geometry::get_coal(nsHP::eMat emat_x) const{
  return get_padless_dim(emat_x, true);
}


bool Geometry::coal_is_pll_k(nsHP::eMat emat_x) const{
  /* proof : false, false, true should give 1 */
  return (static_cast<unsigned>(isColMajor) + static_cast<unsigned>(tX.at(emat_x)) + static_cast<unsigned>(emat_x == nsHP::matA)) % 2;
}


void Geometry::initialise(bool isColMajor_, bool tA_, bool tB_, bool tC_, unsigned lda_, unsigned ldb_, unsigned ldc_, unsigned m_, unsigned n_, unsigned k_, unsigned workspace_size_, char floattype_){
  
  isColMajor = isColMajor_;
  m = m_;
  n = n_;
  k = k_;
  workspace_size = workspace_size_;
  floattype = floattype_;
  
  tX.resize(nsHP::nMats);
  tX[nsHP::matA] = tA_;
  tX[nsHP::matB] = tB_;  
  tX[nsHP::matC] = tC_;

  ldX.resize(nsHP::nMats);
  ldX[nsHP::matA] = lda_;
  ldX[nsHP::matB] = ldb_;  
  ldX[nsHP::matC] = ldc_;
  
  
  if (floattype != 'd' and floattype != 'f'){
    throw  miog_error("floattype should be one of 'f' and 'd' (in Geometry constructor)");
  }
    
  
  check_ldx_consistent();
  
  derived.reset(floattype); 

}
  

Geometry::Geometry(bool isColMajor_, bool tA_, bool tB_, bool tC_, unsigned lda_, unsigned ldb_, unsigned ldc_, unsigned m_, unsigned n_, unsigned k_, unsigned workspace_size_, char floattype_) {

  initialise(isColMajor_, tA_, tB_, tC_, lda_, ldb_, ldc_, m_, n_, k_, workspace_size_, floattype_);


}


std::map<std::string, unsigned> get_key_val_map(std::string geometry_string){
  auto frags = stringutil::split(geometry_string, "_");
  std::map<std::string, unsigned> key_val_map;
  for  (auto & frag : frags){
    auto key_val = stringutil::splitnumeric(frag);
    auto key = std::get<0>(key_val);
    auto val = std::get<1>(key_val);
    key_val_map[key] = val;
  }
  return key_val_map;
}



unsigned safeat(std::map<std::string, unsigned> & map, std::string key){
  if (map.count(key) == 0){
    std::stringstream errm;
    errm << "Unrecognised key `";
    errm << key << "' in safeat of geometry";
    throw miog_error(errm.str());
  }
  return map.at(key);
}

Geometry::Geometry(std::string geometry_string){
  
  auto key_val_map = get_key_val_map(geometry_string);
  
  Geometry goldstandard_geometry (false, false, false, false, 100,100,100,100,100,100,100,'f'); 
  std::string goldstandard_geometry_string = goldstandard_geometry.get_string();
  auto goldstandard_map = get_key_val_map(goldstandard_geometry_string);

  
  std::stringstream errm_ss;
  bool good_string = true;
  for (auto & x : key_val_map){
    if (goldstandard_map.count(x.first) == 0){
      errm_ss << "The key in the geometry string `" << x.first << "' is not valid.  ";
      good_string = false;
    }
  }
  
  for (auto & x : goldstandard_map){
    if (key_val_map.count(x.first) == 0){
      errm_ss << "The geometry string should contain key `" << x.first << "', but does not.  ";
      good_string = false;      
    }
  }

  if (good_string == false){
    throw miog_error(errm_ss.str());
  }
   

  initialise(safeat(key_val_map, "colMaj"), safeat(key_val_map, "tA"), safeat(key_val_map, "tB"), safeat(key_val_map, "tC"), safeat(key_val_map, "lda"), safeat(key_val_map, "ldb"), safeat(key_val_map, "ldc"), safeat(key_val_map, "m"), safeat(key_val_map, "n"), safeat(key_val_map, "k"), safeat(key_val_map, "ws"), get_floattype(safeat(key_val_map, "f")));
}

  

std::string Geometry::get_string() const{
  
  return get_networkconfig_string();

}

std::string Geometry::get_networkconfig_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC" << tX[nsHP::matC]  << "_tA" << tX[nsHP::matA] << "_tB" << tX[nsHP::matB] << "_colMaj" << isColMajor << "_m" << m << "_n" << n << "_k" << k << "_lda" << ldX[nsHP::matA] << "_ldb" << ldX[nsHP::matB] << "_ldc" << ldX[nsHP::matC] << "_ws" << workspace_size << "_f" << derived.float_size_bits;
  return geometry_stringstream.str();
}


}
