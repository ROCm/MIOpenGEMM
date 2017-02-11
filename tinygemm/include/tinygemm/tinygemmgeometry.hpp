#ifndef PROBLEMGEOMETRY_HPP
#define PROBLEMGEOMETRY_HPP

#include <string>

namespace tinygemm{

class TinyGemmOffsets{
  public:
    unsigned oa;
    unsigned ob;
    unsigned oc;
    unsigned oworkspace;
    
    TinyGemmOffsets(unsigned oa, unsigned ob, unsigned oc, unsigned oworkspace);
    
    const unsigned & operator[](char c) const; 
    
};

class TinyGemmGeometryDerived{
public:
  //unsigned dim_c_coal;
  //unsigned dim_c_uncoal;
  unsigned float_size_bits;
  unsigned float_size_bytes;
  
  void reset(char floattype); //bool tC, bool isColMajor, unsigned n, unsigned m, 
  
};

class TinyGemmGeometry{
public:
  /* */
  bool isColMajor; 
  bool tA;
  bool tB;
  bool tC;
  unsigned lda;
  unsigned ldb;
  unsigned ldc;
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
  
  unsigned get_padless_dim(char x, bool isCoal) const;
  
  unsigned get_coal(char x) const;
  
  unsigned get_uncoal(char x) const;
  
  unsigned get_ld(char x) const;
  
  std::string get_string() const;
  
  std::string get_networkconfig_string() const;
  
  /* how far away is gg ? Note that this is not a true distance: it is not symmetrical */
  float get_distance(const TinyGemmGeometry & gg) const;

};


}

#endif
