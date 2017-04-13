#ifndef PROBLEMGEOMETRY_HPP
#define PROBLEMGEOMETRY_HPP

#include <string>
#include <vector>

namespace tinygemm{


namespace nsHP{
  enum eMat {matA, matB, matC, nMats};
}



class TinyGemmOffsets{
  public:
    unsigned oa;
    unsigned ob;
    unsigned oc;
    unsigned oworkspace;
    unsigned tail_off_a;
    unsigned tail_off_b;
    unsigned tail_off_c;

    
    TinyGemmOffsets(unsigned oa, unsigned ob, unsigned oc, unsigned oworkspace, unsigned tail_off_a, unsigned tail_off_b, unsigned tail_off_c);
    
    const unsigned & operator[](char c) const; 
    
};

class TinyGemmGeometryDerived{
public:
  unsigned float_size_bits;
  unsigned float_size_bytes;
  void reset(char floattype);
  
};

class TinyGemmGeometry{
public:
  /* */
  bool isColMajor; 
  
  
  /* indexed by eMat  (for A, B and C) */
  std::vector<bool> tX;
  
  /* indexed by eMat (for A, B and C) */
  std::vector<unsigned> ldX;
      
  unsigned m;
  unsigned n;
  unsigned k; 
  
  unsigned workspace_size;
//* 'f' : 32-bit single precision
//* 'd' : 64-bit double precision 
  const char floattype;
  /* */
  TinyGemmGeometryDerived derived;
  

  TinyGemmGeometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, unsigned workspace_size, char floattype);
  
  TinyGemmGeometry (const TinyGemmGeometry & ) = default;
  
  TinyGemmGeometry & operator= (const TinyGemmGeometry & ) = default;
  
  unsigned get_padless_dim(nsHP::eMat emat_x, bool isCoal) const;
    
  unsigned get_coal(nsHP::eMat emat_x) const;
  
  unsigned get_uncoal(nsHP::eMat emat_x) const;
    
  unsigned get_non_k_dim(nsHP::eMat emat_x) const;
    
  bool coal_is_pll_k(nsHP::eMat emat_x) const;
      
  std::string get_string() const;
  
  std::string get_networkconfig_string() const;
  
  /* how far away is gg ? Note that this is not a true distance: it is not symmetrical */
  float get_distance(const TinyGemmGeometry & gg) const;

};


}

#endif
