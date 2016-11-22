#ifndef SIZING_UP
#define SIZING_UP

#include "problemgeometry.hpp"
namespace sizingup{

/* how many elements in the matrix? Includes padding (when ldx > min possible ldx) 
 * matrix is h x w */
size_t get_n_elements_padded(unsigned h, unsigned w, unsigned ldx, bool isColMajor, bool tX, unsigned offset);

/* check that the strides from first to last addresses are within limits of unsigned. 
 * This is temporary, eventually the code accomodate sufficiently large matrices by casting to size_t  or uint64 when nec.*/
void check_sizes_ok_for_unsigned(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldc, unsigned a_offset, unsigned b_offset, unsigned c_offset);

void check_sizes_ok_for_unsigned(const gemmgeometry::Geometry & gg);

/* set n_work_groups, local_work_size, global_work_size, given m,n, and kernel dimensions (n_work_items_per_c_elm, macro tile size, n_workitems_per_workgroup = local_work_size) */
void set_workforce(size_t & n_work_groups, size_t & local_work_size, size_t & global_work_size, unsigned m, unsigned n, unsigned n_work_items_per_c_elm, unsigned macro_tile_height, unsigned macro_tile_width, unsigned n_workitems_per_workgroup);

  
} //  namespace

#endif
