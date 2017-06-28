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

#ifndef GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP
#define GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <string>
#include <vector>
#include <miopengemm/geometry.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{
namespace dev
{

template <typename TFloat>
void benchgemm(const std::vector<std::string>& hyperstrings,
               unsigned                        n_runs,
               const Geometry&                 gg,
               const Offsets&                  toff,
               const TFloat*                   a,
               const TFloat*                   b,
               const TFloat*                   c,
               outputwriting::OutputWriter&    mowri);

template <typename TFloat>
void accuracy_test(const std::string&           hyperstring,
                   const Geometry&              gg,
                   const Offsets&               toff,
                   const TFloat*                a,
                   const TFloat*                b,
                   const TFloat*                c,
                   const TFloat*                c_true_for_test,
                   outputwriting::OutputWriter& mowri);

template <typename TFloat>
Solution find(const FindParams&            find_params,
              const TFloat*                a,
              const TFloat*                b,
              const TFloat*                c,
              std::string                  constraints_string,
              const Geometry&              gg,
              const Offsets&               toff,
              outputwriting::OutputWriter& mowri);
}
}

#endif
