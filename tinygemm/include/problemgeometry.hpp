#ifndef PROBLEMGEOMETRY_HPP
#define PROBLEMGEOMETRY_HPP

#include <string>

namespace gemmgeometry{

class Geometry{
  public:
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
    unsigned a_offset;
    unsigned b_offset;
    unsigned c_offset; 
    
    Geometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, unsigned a_offset, unsigned b_offset, unsigned c_offset);
    
    Geometry (const Geometry & ) = default;

    Geometry() = default;
    
    std::string get_string() const;
    
    /* how far away is gg ? Note that this is not a true distance: it is not symmetrical */
    float get_distance(const Geometry & gg) const;
    
};


}

#endif
