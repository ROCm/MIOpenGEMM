/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP
#define GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP

#include <memory>
#include <stdlib.h>
#include <string>
#include <vector>
#include <miopengemm/geometry.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/tinyone.hpp>

namespace MIOpenGEMM
{
namespace dev
{

class TinyTwo
{

  private:
  std::unique_ptr<TinyOne<double>> d_moa{nullptr};
  std::unique_ptr<TinyOne<float>>  f_moa{nullptr};
  char                             active_type{'?'};

  template <typename TFloat>
  std::unique_ptr<TinyOne<TFloat>>& get_up_moa()
  {
    throw miog_error("unrecognised template parameter TFloat in TinyTwo get_up_moa");
  }

  template <typename TFloat>
  void set_active_type()
  {
    throw miog_error("unrecognised template parameter TFloat in TinyTwo set_active_type");
  }

  public:
  template <typename TFloat>
  TinyTwo(Geometry        gg_,
          Offsets         toff_,
          const TFloat*   a_,
          const TFloat*   b_,
          const TFloat*   c_,
          owrite::Writer& mowri_,
          const CLHint&   xhint)
  {
    get_up_moa<TFloat>().reset(new TinyOne<TFloat>(gg_, toff_, a_, b_, c_, mowri_, xhint));
    set_active_type<TFloat>();
  }

  TinyTwo(Geometry gg_, Offsets toff_, owrite::Writer& mowri_, const CLHint& xhint);

  std::vector<std::vector<double>> benchgemm(const std::vector<HyPas>& hps, const Halt& hl);

  Solution find2(const FindParams& find_params, const Constraints& constraints);

  // template <typename TFloat>
  // void accuracy_test(const HyPas& hp)
  //{
  // get_up_moa<TFloat>()->accuracy_test(hp);
  //}

  void accuracy_test(const HyPas& hp);
};

template <>
std::unique_ptr<TinyOne<float>>& TinyTwo::get_up_moa<float>();

template <>
std::unique_ptr<TinyOne<double>>& TinyTwo::get_up_moa<double>();

template <>
void TinyTwo::set_active_type<float>();

template <>
void TinyTwo::set_active_type<double>();
}
}

#endif
