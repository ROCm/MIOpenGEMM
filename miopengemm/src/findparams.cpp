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

#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>

namespace MIOpenGEMM
{

std::vector<std::string> get_sumstatkey()
{
  std::vector<std::string> ssv(nSumStatKeys, "unset");
  ssv[Mean]   = "Mean";
  ssv[Median] = "Median";
  ssv[Max]    = "Max";
  for (unsigned i = 0; i < nSumStatKeys; ++i)
  {
    if (ssv[i] == "unset")
    {
      throw miog_error("one of the keys has not been set for sumstatkey");
    }
  }

  return ssv;
}

const std::vector<std::string> sumstatkey = get_sumstatkey();

std::string get_sumstatkey(SummaryStat sumstat)
{

  if (sumstat >= nSumStatKeys)
  {
    throw miog_error("unrecognised sumstat key in get_sumstatkey");
  }
  return sumstatkey[sumstat];
}

FindParams::FindParams(float       allotted_time_,
                       unsigned    allotted_descents_,
                       unsigned    n_runs_per_kernel_,
                       SummaryStat sumstat_)
  : allotted_time(allotted_time_),
    allotted_descents(allotted_descents_),
    n_runs_per_kernel(n_runs_per_kernel_),
    sumstat(sumstat_)
{

  if (allotted_time <= 0)
  {
    throw miog_error("allotted_time should be strictly positive, in FindParams constructor");
  }

  if (allotted_descents == 0)
  {
    throw miog_error("allotted_descents should be strictly positive, in "
                     "FindParams constructor");
  }
}

std::string FindParams::get_string() const
{
  std::stringstream ss;
  ss << "allotted time: " << allotted_time << " allotted_descents: " << allotted_descents
     << " n_runs_per_kernel: " << n_runs_per_kernel << " sumstat: " << get_sumstatkey(sumstat)
     << std::endl;
  return ss.str();
}
}
