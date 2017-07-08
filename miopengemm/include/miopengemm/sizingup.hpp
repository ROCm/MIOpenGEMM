/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_SIZING_UP_HPP
#define GUARD_MIOPENGEMM_SIZING_UP_HPP

#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM
{
namespace sizingup
{

// how many elements in the matrix?
// Includes padding (when ldx > min possible ldx)
// matrix is h x w

size_t get_n_elements_padded(size_t h,
                             size_t w,
                             size_t ldx,
                             bool     isColMajor,
                             bool     tX,
                             size_t offset,
                             size_t tail_off);

// check that the strides from first to last addresses are within limits of size_t.
// This is temporary, eventually the code accomodate sufficiently large matrices by casting to
// size_t  or uint64 when nec
void check_sizes_ok_for_size_t(bool     isColMajor,
                                 bool     tA,
                                 bool     tB,
                                 bool     tC,
                                 size_t m,
                                 size_t n,
                                 size_t k,
                                 size_t lda,
                                 size_t ldb,
                                 size_t ldc,
                                 size_t workspace_size,
                                 size_t a_offset,
                                 size_t b_offset,
                                 size_t c_offset,
                                 size_t workspace_offset,
                                 size_t tail_off_a,
                                 size_t tail_off_b,
                                 size_t tail_off_c);

void check_sizes_ok_for_size_t(const Geometry& gg, const Offsets& toff);

}
}

#endif
