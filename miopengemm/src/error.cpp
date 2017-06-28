/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 *******************************************************************************/

#include <iostream>
#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{

std::string tgformat(const std::string& what_arg, std::string prefix, std::string suffix)
{
  std::stringstream fms;
  fms << "\n\n";
  unsigned    l_terminal = 95;
  auto        frags      = stringutil::split(what_arg, "\n");
  std::string space;
  space.resize(2 * l_terminal, ' ');
  std::string interspace("   ");
  unsigned    line_current = 1;
  for (auto& x : frags)
  {
    if (x.size() == 0)
    {
      x = " ";
    }
    unsigned p_current = 0;
    while (p_current < x.size())
    {
      auto l_current = std::min<unsigned>(l_terminal, x.size() - p_current);
      fms << prefix << interspace << x.substr(p_current, l_current)
          << space.substr(0, l_terminal - l_current) << interspace << suffix << " (" << line_current
          << ")\n";
      p_current += l_current;
      ++line_current;
    }
  }
  fms << "\n";
  return fms.str();
}

miog_error::miog_error(const std::string& what_arg)
  : std::runtime_error(tgformat(what_arg, "MIOpenGEMM", "ERROR"))
{
}

void miog_warning(const std::string& warning)
{
  std::cerr << tgformat(warning, "MIOpenGEMM", "WARNING") << std::flush;
}
}
