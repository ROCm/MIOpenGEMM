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

#ifndef GUARD_MIOPENGEMM_FORALLGENERATOR_HPP
#define GUARD_MIOPENGEMM_FORALLGENERATOR_HPP

#include <miopengemm/prepgenerator.hpp>

namespace MIOpenGEMM
{
namespace bylinegen
{

class ByLineGenerator : public prepgen::PrepGenerator
{

  private:
  unsigned n_full_work_items_per_line;
  unsigned n_work_items_per_line;
  unsigned n_full_work_items;
  unsigned start_in_coal_last_work_item;
  unsigned work_for_last_item_in_coal;

  protected:
  std::string description_string;
  std::string inner_work_string;

  virtual size_t get_work_per_thread() = 0;

  size_t get_n_work_groups() override final;

  public:
  ByLineGenerator(const hyperparams::HyperParams&     hp_,
                  const Geometry&                     gg_,
                  const derivedparams::DerivedParams& dp_,
                  std::string                         type_);

  KernelString get_kernelstring() final override;
  void         setup() final override;

  private:
  void append_description_string(std::stringstream& ss);
  void append_how_definitions(std::stringstream& ss);
  void append_copy_preprocessor(std::stringstream& ss);
  void append_derived_definitions(std::stringstream& ss);

  void append_setup_coordinates(std::stringstream& ss);
  void append_positioning_x_string(std::stringstream& ss);
  void append_inner_work(std::stringstream& ss);
  void append_work_string(std::stringstream& ss);
  void append_positioning_w_string(std::stringstream& ss);

  protected:
  virtual void setup_additional()                                           = 0;
  virtual void append_derived_definitions_additional(std::stringstream& ss) = 0;
};
}
}

#endif
