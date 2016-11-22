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
    
    Geometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k);
    
    Geometry (const Geometry & ) = default;
    
    std::string get_string() const;
    
    /* how far away is gg ? Note that this is not a true distance: it is not symmetrical */
    float distance(const Geometry & gg);
    
};


}

#endif
