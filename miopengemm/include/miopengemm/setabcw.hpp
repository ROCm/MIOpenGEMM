/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_SETABCW_HPP
#define GUARD_MIOPENGEMM_SETABCW_HPP

#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM
{

// for filling matrices with random float values
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

template <typename TFl>
class CpuMemBundle
{
  public:
  std::array<std::vector<TFl>, Mat::E::N> a_mem;
  std::vector<std::vector<TFl>*> v_mem;
  std::array<const TFl*, Mat::E::N> r_mem;

  CpuMemBundle(const std::vector<Geometry>& geoms, const Offsets& offsets)
  {
    v_mem = {&a_mem[Mat::E::A], &a_mem[Mat::E::B], &a_mem[Mat::E::C]};
    setabcw::set_multigeom_abc<TFl>(v_mem, geoms, offsets);
    r_mem = {{a_mem[Mat::E::A].data(), a_mem[Mat::E::B].data(), a_mem[Mat::E::C].data()}};
  }
};
}
}

#endif
