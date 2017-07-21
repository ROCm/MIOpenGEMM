/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_SETABCW_HPP
#define GUARD_MIOPENGEMM_SETABCW_HPP

#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM
{
namespace setabcw
{

template <typename TFloat>
using MatData = std::vector<std::vector<TFloat>*>;

template <typename TFloat>
void set_abc(const MatData<TFloat>& v_abc, const Geometry& gg, const Offsets& toff);

template <typename TFloat>
void set_multigeom_abc(const MatData<TFloat>& v_abc,
                       const std::vector<Geometry>&,
                       const Offsets& toff);

template <typename TFloat>
void set_abcw(const MatData<TFloat>& v_abcw, const Geometry& gg, const Offsets& toff);
}
}

#endif
