/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_REDIRECTION_HPP
#define GUARD_MIOPENGEMM_REDIRECTION_HPP

#include <miopengemm/geometry.hpp>
namespace MIOpenGEMM
{
namespace redirection
{

// convert GEMM problem to equivalent "canonical" format, where isColMajor is true and tC is false

template <typename TFloat>
void redirect(bool&          isColMajor,
              bool&          tA,
              bool&          tB,
              bool&          tC,
              size_t&        m,
              size_t&        n,
              size_t&        lda,
              size_t&        ldb,
              size_t&        a_offset,
              size_t&        b_offset,
              const TFloat*& a,
              const TFloat*& b);

void redirect(bool&        isColMajor,
              bool&        tA,
              bool&        tB,
              bool&        tC,
              size_t&      m,
              size_t&      n,
              std::string& a,
              std::string& b);

void confirm_redirection(bool isColMajor, bool tC);

Geometry get_canonical(const Geometry& gg, bool& swap_ab);

Geometry get_canonical(const Geometry& gg);

bool get_is_not_canonical(const Geometry& gg);
}
}
#endif
