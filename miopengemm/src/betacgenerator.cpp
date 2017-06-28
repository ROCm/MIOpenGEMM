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
#include <miopengemm/betacgenerator.hpp>

namespace MIOpenGEMM
{
namespace betacgen
{

BetacGenerator::BetacGenerator(const hyperparams::HyperParams&     hp_,
                               const Geometry&                     gg_,
                               const derivedparams::DerivedParams& dp_)
  : bylinegen::ByLineGenerator(hp_, gg_, dp_, "betac")
{
}

size_t BetacGenerator::get_local_work_size() { return dp.betac_local_work_size; }

size_t BetacGenerator::get_work_per_thread() { return dp.betac_work_per_thread; }

void BetacGenerator::setup_additional()
{
  initialise_matrixtype('c');

  description_string = R"(
/* ****************************************************
* It is used to perform the beta*C step in GEMM, 
* where recall GEMM has C <- alpha*A*B + beta*C
* It is not quite an axpy, as when ldc is not minimal, 
* C is not contiguous memory  
****************************************************** */ )";
  inner_work_string  = "\n/* the beta scaling */\nc[i] *= beta;";
}

void BetacGenerator::append_derived_definitions_additional(std::stringstream& ss) { ss << " "; }

KernelString get_betac_kernelstring(const hyperparams::HyperParams&     hp,
                                    const Geometry&                     gg,
                                    const derivedparams::DerivedParams& dp)
{
  BetacGenerator bcg(hp, gg, dp);
  bcg.setup();
  return bcg.get_kernelstring();
}
}
}
