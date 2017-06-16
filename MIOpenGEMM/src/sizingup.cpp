#include <stdlib.h>
#include <cmath>
#include <string>
#include <MIOpenGEMM/error.hpp>

#include <MIOpenGEMM/sizingup.hpp>
#include <MIOpenGEMM/geometry.hpp>

namespace MIOpenGEMM{
namespace sizingup{



size_t get_n_elements_padded(unsigned h, unsigned w, unsigned ldx, bool isColMajor, bool tX, unsigned offset, unsigned tail_off){
  size_t nelements = ((unsigned(isColMajor) + unsigned(tX))%2 == 1) ? 
  static_cast<size_t>(ldx)*static_cast<size_t>(w):
  static_cast<size_t>(ldx)*static_cast<size_t>(h);
  nelements += offset;
  nelements += tail_off;
  return nelements;
}


void check_sizes_ok_for_unsigned(const Geometry & gg, const Offsets & toff){
  
  check_sizes_ok_for_unsigned(gg.isColMajor, gg.tX[nsHP::matA], gg.tX[nsHP::matB], gg.tX[nsHP::matC], gg.m, gg.n, gg.k, gg.ldX[nsHP::matA], gg.ldX[nsHP::matB], gg.ldX[nsHP::matC], gg.workspace_size, toff.oa, toff.ob, toff.oc, toff.oworkspace, toff.tail_off_a, toff.tail_off_b, toff.tail_off_c);
  
}

void check_sizes_ok_for_unsigned(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldc, unsigned workspace_size, unsigned a_offset, unsigned b_offset, unsigned c_offset, unsigned workspace_offset, unsigned tail_off_a, unsigned tail_off_b, unsigned tail_off_c){
  
  size_t max_size = std::pow(2, 8*sizeof(unsigned)) - 1;

  std::string base_frag("is too large : unsigned will wrap in address space. Code needs modification. \n");
  std::string errm = "";

  if (sizingup::get_n_elements_padded(m, k, lda, isColMajor, tA, a_offset, tail_off_a)  >= max_size){
    errm += "a";
    errm += base_frag;
  }

  if (sizingup::get_n_elements_padded(k, n, ldb, isColMajor, tB, b_offset, tail_off_b)  >= max_size){
    errm += "b";
    errm += base_frag;
  }

  if (sizingup::get_n_elements_padded(m, n, ldc, isColMajor, tC, c_offset, tail_off_c)  >= max_size){
    errm += "c";
    errm += base_frag;
  }
  
  if (workspace_size + workspace_offset >= max_size){
    errm += "(workspace_size + workspace_offset)";
    errm += base_frag;
    errm += "\nperhaps a smaller workspace_size can be provided?\n";
  }
  
  if (errm.compare("") != 0){
    errm += "\nthis error is can be fixed, just need to change some unsigneds to size_ts. please report this bug"; 
    throw miog_error(errm);
  }
}
  
  
}
}
