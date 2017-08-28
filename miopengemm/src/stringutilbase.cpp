/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <cmath>
#include <iostream>
#include <sstream>
#include <tuple>
#include <miopengemm/error.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{
namespace stringutil
{

void indentify(std::string& source)
{
  std::string newsource;
  newsource.reserve(source.length());
  std::string::size_type last_lend = source.find("\n", 0);

  if (std::string::npos == last_lend)
  {
    std::stringstream errm;

    errm << "the kernel up for indentification seems suspicious: "
         << "it seems like it has no new lines in it :\n"
         << source;
    throw miog_error(errm.str());
  }

  std::string::size_type next_lend  = source.find("\n", last_lend + 1);
  std::string::size_type next_open  = source.find("{", 0);
  std::string::size_type next_close = source.find("}", 0);
  newsource.append(source, 0, last_lend);
  int indent_level = 0;

  while (std::string::npos != next_lend)
  {

    if (next_open < last_lend)
    {
      indent_level += 1;
      next_open = source.find("{", next_open + 1);
    }
    else if (next_close < next_lend)
    {
      indent_level -= 1;
      next_close = source.find("}", next_close + 1);
    }

    else
    {
      newsource.append("\n");
      for (int i = 0; i < indent_level; ++i)
      {
        newsource.append("  ");
      }
      newsource.append(source, last_lend + 1, next_lend - last_lend - 1);
      last_lend = next_lend;
      next_lend = source.find("\n", next_lend + 1);
    }
  }

  newsource += source.substr(last_lend);
  source.swap(newsource);
}

// split the string tosplit by delim.
// With x appearances of delim in tosplit,
// the returned vector will have length x + 1
// (even if appearances at the start, end, contiguous)
std::vector<std::string> split(const std::string& tosplit, const std::string& delim)
{

  // vector to return
  std::vector<std::string> spv;
  if (delim.length() > tosplit.length())
  {
    return spv;
  }

  std::vector<size_t> splitposstarts{0};
  std::vector<size_t> splitposends;

  for (size_t x = 0; x < tosplit.length() - delim.length() + 1; ++x)
  {
    auto res = std::mismatch(delim.begin(), delim.end(), tosplit.begin() + x);
    if (res.first == delim.end())
    {
      splitposends.push_back(x);
      splitposstarts.push_back(x + delim.length());
    }
  }

  splitposends.push_back(tosplit.length());

  for (size_t i = 0; i < splitposends.size(); ++i)
  {
    spv.push_back(tosplit.substr(splitposstarts[i], splitposends[i] - splitposstarts[i]));
  }

  return spv;
}

std::tuple<std::string, size_t> splitnumeric(std::string alphanum)
{
  size_t split_point = alphanum.find_first_of("0123456789");

  if (split_point == std::string::npos)
  {
    throw miog_error("This error is being thrown from stringutilbase.cpp, "
                     "function splitnumeric. "
                     "It seems like the input string `" +
                     alphanum + "' has no digits in it.");
  }
  return std::make_tuple<std::string, size_t>(
    alphanum.substr(0, split_point), std::stoi(alphanum.substr(split_point, alphanum.size())));
}

bool isws(const char& c) { return (c == ' ' || c == '\t' || c == '\n'); }

std::vector<std::string> split(const std::string& tosplit)
{

  std::vector<std::string> spv2;

  size_t it{0};

  while (it != tosplit.size())
  {
    while (isws(tosplit[it]) and it != tosplit.size())
    {
      ++it;
    }
    size_t start = it;

    while (!isws(tosplit[it]) and it != tosplit.size())
    {
      ++it;
    }
    size_t end = it;

    if (!isws(tosplit[end - 1]))
    {
      spv2.push_back(tosplit.substr(start, end - start));
    }
  }

  return spv2;
}

std::string getdirfromfn(const std::string& fn)
{
  auto morcels = split(fn, "/");

  if (morcels[0].compare("") != 0)
  {
    throw miog_error("The string passed to getdirfromfn is not a valid path as "
                     "there is no leading / .");
  }

  std::string dir = "/";

  for (size_t i = 1; i < morcels.size() - 1; ++i)
  {
    dir = dir + morcels[i] + "/";
  }

  return dir;
}

std::string get_padded(size_t x, size_t length)
{
  size_t x_length = 0;
  if (x < 10)
  {
    x_length = 1;
  }
  else if (x < 100)
  {
    x_length = 2;
  }
  else if (x < 1000)
  {
    x_length = 3;
  }
  else if (x < 10000)
  {
    x_length = 4;
  }
  else
  {
    x_length = size_t(std::log10(x + 1));
  }

  auto        n_pads = length + 1 - x_length;
  std::string padded = std::to_string(x);
  for (auto sp = 0; sp < n_pads; ++sp)
  {
    padded = padded + ' ';
  }
  return padded;
}

std::string get_stars(size_t n_stars) { return std::string(n_stars, '*'); }

std::string get_star_wrapped(const std::string& s)
{
  auto              n_stars = s.size() + 1;
  std::stringstream ss;
  ss << "/* " << get_stars(n_stars) << "\n* " << s << " *\n" << get_stars(n_stars) << "  */";
  return ss.str();
}

void add_v_string(std::stringstream& ss, const std::vector<size_t>& values)
{
  ss << " { ";
  for (auto& x : values)
  {
    ss << x << ' ';
  }
  ss << "}\n";
}
}
}
