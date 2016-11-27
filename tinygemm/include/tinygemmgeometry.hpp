#ifndef PROBLEMGEOMETRY_HPP
#define PROBLEMGEOMETRY_HPP

#include <string>

namespace tinygemm{

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
    unsigned a_offset;
    unsigned b_offset;
    unsigned c_offset; 
    /* */
    
    TinyGemmGeometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k, unsigned a_offset, unsigned b_offset, unsigned c_offset);
    
    TinyGemmGeometry (const TinyGemmGeometry & ) = default;

    TinyGemmGeometry() = default;
    
    std::string get_string() const;

    std::string get_networkconfig_string() const;
    
    /* how far away is gg ? Note that this is not a true distance: it is not symmetrical */
    float get_distance(const TinyGemmGeometry & gg) const;
    
};


}

#endif
