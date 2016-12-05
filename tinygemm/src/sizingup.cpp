#include <stdlib.h>
#include <cmath>
#include <string>
#include <tinygemm/tinygemmerror.hpp>

#include <tinygemm/sizingup.hpp>
#include <tinygemm/tinygemmgeometry.hpp>

namespace tinygemm{
namespace sizingup{



void set_workforce(size_t & n_work_groups, size_t & local_work_size, size_t & global_work_size, unsigned m, unsigned n, unsigned n_work_items_per_c_elm, unsigned macro_tile_height, unsigned macro_tile_width, unsigned n_workitems_per_workgroup){    
  n_work_groups = n_work_items_per_c_elm * ((m/macro_tile_height) + (m%macro_tile_height != 0)) * ((n/macro_tile_width) + (n%macro_tile_width != 0));
  local_work_size = n_workitems_per_workgroup;
  global_work_size = n_work_groups * local_work_size;
}


size_t get_n_elements_padded(unsigned h, unsigned w, unsigned ldx, bool isColMajor, bool tX, unsigned offset){
  size_t nelements = ((unsigned(isColMajor) + unsigned(tX))%2 == 1) ? 
  static_cast<size_t>(ldx)*static_cast<size_t>(w):
  static_cast<size_t>(ldx)*static_cast<size_t>(h);
  nelements += offset;
  return nelements;
}


void check_sizes_ok_for_unsigned(const tinygemm::TinyGemmGeometry & gg){
  check_sizes_ok_for_unsigned(gg.isColMajor, gg.tA, gg.tB, gg.tC, gg.m, gg.n, gg.k, gg.lda, gg.ldb, gg.ldc, gg.a_offset, gg.b_offset, gg.c_offset);
}

void check_sizes_ok_for_unsigned(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldc, unsigned a_offset, unsigned b_offset, unsigned c_offset){
  
  size_t max_size = std::pow(2, 8*sizeof(unsigned));

  std::string base_frag("is too large : unsigned will wrap in address space. Code needs modification. \n");
  std::string errm = "";

  if (sizingup::get_n_elements_padded(m, k, lda, isColMajor, tA, a_offset)  >= max_size){
    errm += "a";
    errm += base_frag;
  }

  if (sizingup::get_n_elements_padded(k, n, ldb, isColMajor, tB, b_offset)  >= max_size){
    errm += "b";
    errm += base_frag;
  }

  if (sizingup::get_n_elements_padded(m, n, ldc, isColMajor, tC, c_offset)  >= max_size){
    errm += "c";
    errm += base_frag;
  }
  
  if (errm.compare("") != 0){
    throw tinygemm_error(errm);
  }
}
  
  
}
}
