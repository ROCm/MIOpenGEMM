#ifndef SIZING_UP
#define SIZING_UP

#include <MIOpenGEMM/geometry.hpp>
namespace MIOpenGEMM{
namespace sizingup{

/* how many elements in the matrix? Includes padding (when ldx > min possible ldx) 
 * matrix is h x w */
size_t get_n_elements_padded(unsigned h, unsigned w, unsigned ldx, bool isColMajor, bool tX, unsigned offset, unsigned tail_off);

/* check that the strides from first to last addresses are within limits of unsigned. 
 * This is temporary, eventually the code accomodate sufficiently large matrices by casting to size_t  or uint64 when nec.*/
void check_sizes_ok_for_unsigned(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, unsigned lda, unsigned ldb, unsigned ldc, unsigned workspace_size, unsigned a_offset, unsigned b_offset, unsigned c_offset, unsigned workspace_offset, unsigned tail_off_a, unsigned tail_off_b, unsigned tail_off_c);

void check_sizes_ok_for_unsigned(const Geometry & gg, const Offsets & toff);
  
} //  namespace
}

#endif
