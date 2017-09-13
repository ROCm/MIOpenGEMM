/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro devices, Inc. All rights reserved.
 *******************************************************************************/

#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/hint.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/platform.hpp>

namespace MIOpenGEMM
{

const size_t hintless = std::numeric_limits<size_t>::max();

bool BasicHint::has_id() const { return id != hintless; }

size_t BasicHint::get_id() const
{
  if (!has_id())
  {
    throw miog_error("should not call get_id unless has_id() confirmed");
  }
  return id;
}

// Matches defaults to "" (essentially *)
BasicHint::BasicHint() : id(hintless), matches({""}) {}

BasicHint::BasicHint(size_t id_, const std::vector<std::string>& matches_)
  : id(id_), matches(matches_)
{
}

bool BasicHint::is_match_hit(const std::string& s2) const
{
  for (auto& x : matches)
  {
    if (s2.find(x) != std::string::npos)
    {
      return true;
    }
  }
  return false;
}

void BasicHint::set(size_t& x, const std::vector<std::string> ts) const
{

  std::stringstream matches_ss;
  matches_ss << "[ ";
  for (auto& fr : matches)
  {
    matches_ss << '`' << fr << "' ";
  }
  matches_ss << "] ";
  std::string matches_string = matches_ss.str();

  std::stringstream xss;
  size_t            n_potentials = ts.size();
  xss << "\nStrings of " << get_description() << " :\n";
  for (size_t i = 0; i < n_potentials; ++i)
  {
    xss << "\nAt index : (" << i << ") ------>\n";
    xss << ts[i];
    xss << '\n';
  }
  std::string potstring = xss.str();

  if (has_id())
  {
    if (get_id() >= n_potentials)
    {
      std::stringstream errm;
      errm << "provided index exceeds number of " << get_description() << ". " << potstring;
      throw miog_error(errm.str());
    }
    else
    {
      x = get_id();
      return;
    }
  }

  else
  {
    std::vector<size_t> pot_indices;
    for (size_t i = 0; i < n_potentials; ++i)
    {
      for (auto& fr : matches)
      {
        if (ts[i].find(fr) != std::string::npos)
        {
          pot_indices.push_back(i);
          break;
        }
      }
    }

    if (pot_indices.size() == 0)
    {
      std::stringstream errm;
      errm << "no potential " << get_description() << " conditional on the hint matches vector ";
      errm << matches_string;
      errm << potstring;
      throw miog_error(errm.str());
    }

    else if (pot_indices.size() > 1)
    {
      std::stringstream errm;
      errm << "multiple " << get_description() << " [";
      for (auto& ppi : pot_indices)
      {
        errm << ' ' << ppi << ' ';
      }
      errm << "] satisfy the hint matches vector ";
      errm << ". ";
      errm << " Refine the hint string vector or directly provide the index.";
      errm << potstring;
      throw miog_error(errm.str());
    }
    else
    {
      x = pot_indices[0];
    }
  }
}

CLHint::CLHint(const std::vector<std::string>& matches)
  : pla(hintless, matches), dev(hintless, matches)
{
}
CLHint::CLHint(size_t pla_id, size_t dev_id) : pla(pla_id, {}), dev(dev_id, {}) {}
}
