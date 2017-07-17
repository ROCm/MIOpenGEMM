/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_HYPERPARAMS_HPP
#define GUARD_MIOPENGEMM_HYPERPARAMS_HPP

#include <array>
#include <functional>
#include <map>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/oclutil.hpp>

namespace MIOpenGEMM
{

std::vector<size_t> get_hy_v(std::string hy_s, bool hy_s_full, Mat::E emat);
std::string get_str(Mat::E emat, const std::vector<size_t>& vs);

class Constraint
{
  public:
  Mat::E              emat;
  std::vector<size_t> range;
  std::vector<size_t> start_range;
  Constraint() = default;
  Constraint(Mat::E);
  Constraint(Mat::E, const std::string& r);
  Constraint(Mat::E, const std::string& r, const std::string& sr);
  std::string get_r_str() const;
  std::string get_sr_str() const;
};

class Constraints
{
  public:
  using str_array = std::array<std::string, Mat::E::N>;
  std::array<Constraint, Mat::E::N> sub;
  Constraints(const str_array& r);
  Constraints(const std::string& rconcat);
  Constraints(const str_array& r, const str_array& sr);
  std::string get_combo_str(const str_array&) const;
  std::string get_r_str() const;
  std::string get_sr_str() const;
};

class SuHy
{
  public:
  using str_array = std::array<std::string, Mat::E::N>;
  Mat::E              emat;
  std::vector<size_t> vs;
  std::string         get_string() const;
  bool operator==(const SuHy& rhs) const;
  void replace_where_defined(const Constraint& constraint);
  void checks() const;
  SuHy() = default;
  SuHy(Mat::E);
  SuHy(Mat::E, const std::string&);
  SuHy(Mat::E, std::vector<size_t>&& vs);
};

class HyPas
{
  public:
  std::array<SuHy, Mat::E::N> sus;
  
  using str_array = std::array<std::string, Mat::E::N>;
  HyPas() = default;
  HyPas(const str_array&);
  HyPas(const std::string&);
  HyPas(std::array<SuHy, Mat::E::N>&&);
  HyPas(const HyPas&) = default;
  void replace_where_defined(const Constraints& constraints);
  std::string get_string() const;
  bool operator==(const HyPas& rhs) const;
  void checks() const;
  bool is_different(KType::E e_ktype, const HyPas & rhs); 
};


//HyPas get_null_hp();

}

#endif
