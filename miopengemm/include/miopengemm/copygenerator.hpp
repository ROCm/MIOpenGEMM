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

#ifndef GUARD_MIOPENGEMM_COPYGENERATOR_HPP
#define GUARD_MIOPENGEMM_COPYGENERATOR_HPP

#include <miopengemm/bylinegenerator.hpp>

namespace MIOpenGEMM
{
namespace copygen
{

class CopyGenerator : public bylinegen::ByLineGenerator
{

  public:
  CopyGenerator(const hyperparams::HyperParams&     hp_,
                const Geometry&                     gg_,
                const derivedparams::DerivedParams& dp_,
                const std::string&                  type_);

  virtual void setup_additional() override final;

  virtual void append_derived_definitions_additional(std::stringstream& ss) override final;

  size_t get_local_work_size() override final;

  size_t get_work_per_thread() override final;
};

KernelString get_copya_kernelstring(const hyperparams::HyperParams&     hp,
                                    const Geometry&                     gg,
                                    const derivedparams::DerivedParams& dp);

KernelString get_copyb_kernelstring(const hyperparams::HyperParams&     hp,
                                    const Geometry&                     gg,
                                    const derivedparams::DerivedParams& dp);
}
}

#endif
