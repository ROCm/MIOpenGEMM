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

#ifndef GUARD_MIOPENGEMM_KERNELSTRINGS_HPP
#define GUARD_MIOPENGEMM_KERNELSTRINGS_HPP

#include <string>
#include <vector>

namespace MIOpenGEMM
{

enum bkt
{
  wsa = 0,
  wsb,
  betac,
  main,
  nBasicKernelTypes
};
// maps bkt to a string
extern const std::vector<std::string> basic_kernel_type_strings;

// maps dependencies of execution order of kernels
extern const std::vector<std::vector<unsigned>> kernel_dependencies;

class KernelType
{

  public:
  // summary of uses_a, uses_b, uses_c etc
  std::string full;

  // one of wsa, wsb, betac, main
  std::string bkt_string;

  bkt basic_kernel_type;

  bool uses_a;
  bool uses_b;
  bool uses_c;
  bool uses_workspace;
  bool uses_alpha;
  bool uses_beta;

  bool uses(char c) const;

  KernelType(bool uses_a_,
             bool uses_b_,
             bool uses_c_,
             bool uses_workspace_,
             bool uses_alpha_,
             bool uses_beta_);

  KernelType() = default;
};

class KernelString
{
  public:
  // type : betac_alphab, betac_workspace, etc
  KernelType  type;
  std::string kernstr;
  std::string fname;

  size_t global_work_size;
  size_t local_work_size;

  KernelString(const KernelType&  type_,
               std::string&&      kernstr_,
               const std::string& fname_,
               size_t             global_work_size_,
               size_t             local_work_size_)
    : type(type_),
      kernstr(kernstr_),
      fname(fname_),
      global_work_size(global_work_size_),
      local_work_size(local_work_size_)
  {
  }

  KernelString() = default;
};
}

#endif
