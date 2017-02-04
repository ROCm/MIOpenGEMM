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
};

class TinyGemmGeometryDerived{
public:
  unsigned dim_c_coal;
  unsigned dim_c_uncoal;
  unsigned float_size_bits;
  unsigned float_size_bytes;
  
  void reset(bool tC, bool isColMajor, unsigned n, unsigned m, char floattype);
  
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
//* the user must guarantee that a, b and c are in agreement with floattype, 
//* TODO is there a way to check float type from a,b,c? If so, floattype is not nec. */
//const char floattype,
  char floattype;
  /* */
  TinyGemmGeometryDerived derived;
  

  TinyGemmGeometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, unsigned workspace_size, char floattype);
  
  TinyGemmGeometry (const TinyGemmGeometry & ) = default;
  
  TinyGemmGeometry & operator= (const TinyGemmGeometry & ) = default;
  
  //TODO : is this used? dangerous.
  //TinyGemmGeometry() = default;
  
  std::string get_string() const;
  
  std::string get_networkconfig_string() const;
  
  /* how far away is gg ? Note that this is not a true distance: it is not symmetrical */
  float get_distance(const TinyGemmGeometry & gg) const;

};


}

#endif
