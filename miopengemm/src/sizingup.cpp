/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <cmath>
#include <stdlib.h>
#include <string>
#include <limits>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/sizingup.hpp>

namespace MIOpenGEMM
{
namespace sizingup
{

size_t get_n_elements_padded(size_t h,
                             size_t w,
                             size_t ldx,
                             bool     isColMajor,
                             bool     tX,
                             size_t offset,
                             size_t tail_off)
{
  size_t nelements = ((size_t(isColMajor) + size_t(tX)) % 2 == 1)
                       ? static_cast<size_t>(ldx) * static_cast<size_t>(w)
                       : static_cast<size_t>(ldx) * static_cast<size_t>(h);
  nelements += offset;
  nelements += tail_off;
  return nelements;
}

void check_sizes_ok_for_size_t(const Geometry& gg, const Offsets& toff)
{

  check_sizes_ok_for_size_t(gg.isColMajor,
                              gg.tX[Mat::E::A],
                              gg.tX[Mat::E::B],
                              gg.tX[Mat::E::C],
                              gg.m,
                              gg.n,
                              gg.k,
                              gg.ldX[Mat::E::A],
                              gg.ldX[Mat::E::B],
                              gg.ldX[Mat::E::C],
                              gg.workspace_size,
                              toff.oa,
                              toff.ob,
                              toff.oc,
                              toff.oworkspace,
                              toff.tail_off_a,
                              toff.tail_off_b,
                              toff.tail_off_c);
}

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
                                 size_t tail_off_c)
{

  size_t max_size = std::numeric_limits<size_t>::max() / 10; 

  std::string base_frag("is too large : `size_t' will wrap in address space (old warning from unsigned time "
                        "Code needs modification. \n");
  std::string errm = "";

  if (sizingup::get_n_elements_padded(m, k, lda, isColMajor, tA, a_offset, tail_off_a) >= max_size)
  {
    errm += "a";
    errm += base_frag;
  }

  if (sizingup::get_n_elements_padded(k, n, ldb, isColMajor, tB, b_offset, tail_off_b) >= max_size)
  {
    errm += "b";
    errm += base_frag;
  }

  if (sizingup::get_n_elements_padded(m, n, ldc, isColMajor, tC, c_offset, tail_off_c) >= max_size)
  {
    errm += "c";
    errm += base_frag;
  }

  if (workspace_size + workspace_offset >= max_size)
  {
    errm += "(workspace_size + workspace_offset)";
    errm += base_frag;
    errm += "\nperhaps a smaller workspace_size can be provided?\n";
  }

  if (errm.compare("") != 0)
  {
    errm += "\nthis error is can be fixed, just need to change some size_ts "
            "to size_ts. please "
            "report this bug";
    throw miog_error(errm);
  }
}
}
}
