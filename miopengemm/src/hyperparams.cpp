/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <algorithm>
#include <sstream>
#include <miopengemm/architests.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/macgrid.hpp>
#include <miopengemm/randomutil.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{

// TODO : Constraints and Hyperparams are very similar, consider inheritance

void reflect_c(std::vector<size_t>& cvs)
{

  if (cvs.size() != NonChi::E::N)
  {
    throw miog_error("cvs should be of size NonChi::E::N, it is " + std::to_string(cvs.size()));
  }

  if (cvs[NonChi::E::SKW] != Status::E::UNDEFINED && cvs[NonChi::E::MAC] != Status::E::UNDEFINED)
  {
    macgrid::Grid grid(cvs[NonChi::E::MAC], cvs[NonChi::E::SKW]);
    if (grid.is_good == false)
    {
      throw miog_error("bad grid in reflect " + grid.error_message);
    }

    cvs[NonChi::E::SKW] = 2 * macgrid::skew0 - cvs[NonChi::E::SKW];
    if (!macgrid::mac_is_square(cvs[NonChi::E::MAC]))
    {
      cvs[NonChi::E::SKW] += 1;  // sum to 2*skew0 + 1
    }
  }

  switch (cvs[NonChi::E::GAL])
  {
  case GroupAllocation::E::BYROW: cvs[NonChi::E::GAL] = GroupAllocation::E::BYCOL; break;
  case GroupAllocation::E::BYCOL: cvs[NonChi::E::GAL] = GroupAllocation::E::BYROW; break;
  default: break;
  }

  switch (cvs[NonChi::E::AFI])
  {
  case Binary::E::YES: cvs[NonChi::E::AFI] = Binary::E::NO; break;
  case Binary::E::NO: cvs[NonChi::E::AFI]  = Binary::E::YES; break;
  default: break;
  }

  switch (cvs[NonChi::E::MIA])
  {
  case MicroAllocation::E::BYA: cvs[NonChi::E::MIA] = MicroAllocation::E::BYB; break;
  case MicroAllocation::E::BYB: cvs[NonChi::E::MIA] = MicroAllocation::E::BYA; break;
  default: break;
  }
}

HyPas HyPas::get_reflected(bool swap_ab) const
{

  if (!swap_ab)
  {
    return *this;
  }

  else
  {

    auto suhyc = sus[Mat::E::C];
    reflect_c(suhyc.vs);
    return {{{sus[Mat::E::B], sus[Mat::E::A], suhyc}}};
  }
}

Constraints Constraints::get_reflected(bool swap_ab) const
{

  Constraints reflected(*this);

  if (swap_ab)
  {
    std::swap(reflected.sub[Mat::E::A], reflected.sub[Mat::E::B]);
    reflect_c(reflected.sub[Mat::E::C].range);
    reflect_c(reflected.sub[Mat::E::C].start_range);
    return reflected;
  }

  return reflected;
}

void SuHy::checks() const
{
  if (vs.size() != Mat::mat_to_xchi(emat)->N)
  {
    throw miog_error("size of vs array of SuHy is not as expected, internal logic error");
  }

  for (const auto& v : vs)
  {
    if (v == Status::E::UNDEFINED)
    {
      throw miog_error("UNDEFINED in vs of SuHy, internal logic error");
    }
  }

  // some specific checks?
}

void HyPas::checks() const
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sus[emat].checks();
  }
}

bool SuHy::operator==(const SuHy& rhs) const { return vs == rhs.vs; }

bool HyPas::operator==(const HyPas& rhs) const { return sus == rhs.sus; }

std::string HyPas::get_string() const
{

  std::stringstream ss;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    if (emat != Mat::E::A)
    {
      ss << "\", ";
    }
    ss << "\"" << sus[emat].get_string();
  }
  ss << "\"";
  return ss.str();
}

std::string Constraints::get_combo_str(const str_array& strs) const
{
  std::stringstream ss;
  bool              empty = true;
  for (auto x : strs)
  {
    if (x.size() > 2)
    {
      if (!empty)
      {
        ss << "__";
      }
      ss << x;
      empty = false;
    }
  }
  return ss.str();
}

std::string Constraints::get_r_str() const
{
  str_array strs;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    strs[emat] = Mat::M().name[emat] + std::string("_") + sub[emat].get_r_str();
  }
  return get_combo_str(strs);
}

std::string Constraints::get_sr_str() const
{
  str_array strs;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    strs[emat] = Mat::M().name[emat] + std::string("_") + sub[emat].get_sr_str();
  }
  return get_combo_str(strs);
}

std::string Constraints::get_string() const { return get_r_str(); }

void SuHy::replace_where_defined(const Constraint& constraint)
{
  if (constraint.emat != emat)
  {
    throw miog_error("constraint is not for same subgraph, internal logic error");
  }
  for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {
    if (constraint.range[i] != Status::E::UNDEFINED)
    {
      vs[i] = constraint.range[i];
    }
  }
}

std::string get_str(Mat::E emat, const std::vector<size_t>& vs)
{
  std::stringstream ss;
  bool              isempty = true;
  for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {
    if (vs[i] != Status::E::UNDEFINED)
    {
      if (!isempty)
      {
        ss << '_';
      }
      ss << Mat::mat_to_xchi(emat)->name[i] << vs[i];
      isempty = false;
    }
  }
  return ss.str();
}

std::string Constraint::get_r_str() const { return get_str(emat, range); }

std::string Constraint::get_sr_str() const { return get_str(emat, start_range); }

std::string SuHy::get_string() const { return get_str(emat, vs); }

Constraint::Constraint(Mat::E e)
  : emat(e),
    range(Mat::mat_to_xchi(emat)->N, Status::E::UNDEFINED),
    start_range(Mat::mat_to_xchi(emat)->N, Status::E::UNDEFINED)
{
}

Constraint::Constraint(Mat::E e, const std::string& r) : Constraint(e)
{
  range = get_hy_v(r, false, emat);
}

Constraint::Constraint(Mat::E e, const std::string& r, const std::string& sr) : Constraint(e, r)
{
  start_range = get_hy_v(sr, false, emat);
}

Constraints::Constraints(const str_array& cr_strings)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sub[emat] = Constraint(emat, cr_strings[emat]);
  }
}

Constraints::Constraints(const str_array& cr, const str_array& csr)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sub[emat] = Constraint(emat, cr[emat], csr[emat]);
  }
}

// included for deprecation reasons
std::array<std::string, Mat::E::N> get_substrings(const std::string& rconcat)
{

  std::array<std::string, Mat::E::N> substrings;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    substrings[emat] = "";
  }

  auto              megafrags = stringutil::split(rconcat, "__");
  std::stringstream ss;
  for (auto& megafrag : megafrags)
  {
    if (Mat::M().val.count(megafrag[0]) == 0)
    {
      ss << "\nWhile reading hyperstring in get-params-from-string,"
         << "the leading char, `" << megafrag[0] << "', was not recognised. "
         << "The fragment in question is " << megafrag << '.';
      throw miog_error(ss.str());
    }
    Mat::E emat    = static_cast<Mat::E>(Mat::M().val.at(megafrag[0]));
    size_t minsize = std::string("m__hv").size();

    if (megafrag.size() < minsize)
    {
      ss << "sub constraint " << megafrag << " is too short, something is wrong. \n";
      throw miog_error(ss.str());
    }
    substrings[emat] = megafrag.substr(2);
  }

  return substrings;
}

Constraints::Constraints(const std::string& rconcat) : Constraints(get_substrings(rconcat)) {}

HyPas::HyPas(const std::string& rconcat) : HyPas(get_substrings(rconcat)) {}

std::vector<size_t> get_hy_v(std::string hy_s, bool hy_s_full, Mat::E emat)
{

  auto p_kv = Mat::mat_to_xchi(emat);

  std::vector<size_t> hy_v(p_kv->N, Status::E::UNDEFINED);

  std::vector<std::string> keyvalfrags;
  if (hy_s.compare(""))
  {
    keyvalfrags = stringutil::split(hy_s, "_");
  }

  // MIC, etc
  std::string key;
  // 6, etc
  size_t val;

  auto start = p_kv->name.begin();
  auto end   = p_kv->name.end();
  for (auto& x : keyvalfrags)
  {
    std::tie(key, val) = stringutil::splitnumeric(x);
    if (std::find(start, end, key) == end)
    {
      std::stringstream ss;
      ss << "While processing the constraint string for Sub Graph `" << Mat::M().name[emat]
         << "', ";
      ss << "the unrecognised key `" + key << "' was not encountered. \n";
      throw miog_error(ss.str());
    }

    size_t keyindex = p_kv->val.at(key);

    if (keyindex >= p_kv->N)
    {
      throw miog_error("keyindex exceeds number of sub graph hyper params, internal logic error ");
    }

    hy_v[keyindex] = val;
  }

  // A special test in the case that constraints
  // are supposed to be comprehensive
  if (hy_s_full == true)
  {
    for (size_t hpi = 0; hpi < p_kv->N; ++hpi)
    {
      if (hy_v[hpi] == Status::E::UNDEFINED)
      {
        std::stringstream ss;
        ss << "While processing the constraints string of SubG `" << Mat::M().name[emat] << "', ";
        ss << "the parameter `" << p_kv->name[hpi]
           << "' appeared to be unset. Values must all be set as "
           << "hy_s_full is true ";
        throw miog_error(ss.str());
      }
    }
  }

  return hy_v;
}

void HyPas::replace_where_defined(const Constraints& constraints)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sus[emat].replace_where_defined(constraints.sub[emat]);
  }
}

SuHy::SuHy(Mat::E e) : emat(e), vs(Mat::mat_to_xchi(emat)->N, Status::E::UNDEFINED) {}

SuHy::SuHy(Mat::E e, const std::string& hyperstring) : SuHy(e)
{
  vs = get_hy_v(hyperstring, true, emat);
  checks();
}

SuHy::SuHy(Mat::E e, std::vector<size_t>&& vs_) : emat(e), vs(vs_) {}

HyPas::HyPas(const str_array& hyperstrings)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sus[emat] = SuHy(emat, hyperstrings[emat]);
  }
}

HyPas::HyPas(std::array<SuHy, Mat::E::N>&& suhys) : sus(suhys) {}
}
