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

#ifndef GUARD_MIOPENGEMM_KERNELCACHE_HPP
#define GUARD_MIOPENGEMM_KERNELCACHE_HPP

#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{

std::string get_cache_keys_string(std::string k_dev,
                                  std::string k_con,
                                  std::string k_geo,
                                  std::string k_comment);

class CachedSolution
{
  public:
  std::string        hyperstring;
  SolutionStatistics stats;
  CachedSolution(std::string hyperstring_, SolutionStatistics stats_)
    : hyperstring(hyperstring_), stats(stats_)
  {
  }
  CachedSolution() = default;

  std::string get_string();
};

/* TODO : unordered maps are faster */
using KernelCache =
  std::map<std::string,
           std::map<std::string, std::map<std::string, std::map<std::string, CachedSolution>>>>;

KernelCache get_kernel_cache();

CachedSolution get_generic_cached_solution(const std::string& constraints_string,
                                           const Geometry&    gg);

// [device][constraint][geometry][further_comment] -> cached solution
extern const KernelCache kernel_cache;

void add_entry(KernelCache&       kc,
               const std::string& k_dev,
               const std::string& k_con,
               const std::string  k_geo,
               const std::string  k_comment,
               CachedSolution     tgcs);
}

#endif
