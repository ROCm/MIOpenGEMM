/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP
#define GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP

#include <memory>
#include <stdlib.h>
#include <string>
#include <vector>
#include <miopengemm/diva.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{
namespace dev
{

class Boa
{

  private:
  std::unique_ptr<Diva<double>> d_moa{nullptr};
  std::unique_ptr<Diva<float>>  f_moa{nullptr};
  char                          active_type{'?'};

  template <typename TFloat>
  std::unique_ptr<Diva<TFloat>>& get_up_moa()
  {
    throw miog_error("unrecognised template parameter TFloat in Boa get_up_moa");
  }

  template <typename TFloat>
  void set_active_type()
  {
    throw miog_error("unrecognised template parameter TFloat in Boa set_active_type");
  }

  public:
  template <typename TFloat>
  Boa(Geometry        gg_,
      Offsets         toff_,
      const TFloat*   a_,
      const TFloat*   b_,
      const TFloat*   c_,
      owrite::Writer& mowri_)
  {
    get_up_moa<TFloat>().reset(new Diva<TFloat>(gg_, toff_, a_, b_, c_, mowri_));
    set_active_type<TFloat>();
  }

  Boa(Geometry gg_, Offsets toff_, owrite::Writer& mowri_);

  void benchgemm(const std::vector<std::string>& hyperstrings, const Halt & hl);

  Solution find(const FindParams& find_params, std::string constraints_string);

  template <typename TFloat>
  void accuracy_test(const std::string& hyperstring, const TFloat* c_true_for_test)
  {
    get_up_moa<TFloat>->accuracy_test(hyperstring, c_true_for_test);
  }

  void accuracy_test(const std::string& hyperstring);
};

template <>
std::unique_ptr<Diva<float>>& Boa::get_up_moa<float>();

template <>
std::unique_ptr<Diva<double>>& Boa::get_up_moa<double>();

template <>
void Boa::set_active_type<float>();

template <>
void Boa::set_active_type<double>();
}
}

#endif
