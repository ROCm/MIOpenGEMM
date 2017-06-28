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

#ifndef GUARD_MIOPENGEMM_PREPGENERATOR_HPP
#define GUARD_MIOPENGEMM_PREPGENERATOR_HPP

#include <miopengemm/basegenerator.hpp>

namespace MIOpenGEMM
{
namespace prepgen
{

class PrepGenerator : public basegen::BaseGenerator
{

  protected:
  unsigned n_work_items;
  unsigned n_work_groups;

  char       matrixchar;
  char       MATRIXCHAR;
  nsHP::eMat emat_x;

  void set_usage_from_matrixchar();
  void append_basic_what_definitions(std::stringstream& ss);

  virtual size_t get_local_work_size() = 0;
  virtual size_t get_n_work_groups()   = 0;

  size_t get_global_work_size()
  {
    size_t forall_global_work_size = get_n_work_groups() * get_local_work_size();
    return forall_global_work_size;
  }

  void initialise_matrixtype(char matrixchar_in)
  {
    if (matrixchar_in == 'a')
    {
      matrixchar = 'a';
      MATRIXCHAR = 'A';
      emat_x     = nsHP::matA;
    }

    else if (matrixchar_in == 'b')
    {
      matrixchar = 'b';
      MATRIXCHAR = 'B';
      emat_x     = nsHP::matB;
    }

    else if (matrixchar_in == 'c')
    {
      matrixchar = 'c';
      MATRIXCHAR = 'C';
      emat_x     = nsHP::matC;
    }

    else
    {
      throw miog_error("in PrepGenerator : unrecognised matrixtype " +
                       std::to_string(matrixchar_in));
    }
  }

  public:
  PrepGenerator(const hyperparams::HyperParams&     hp_,
                const Geometry&                     gg_,
                const derivedparams::DerivedParams& dp_,
                std::string                         type_);
};
}
}
#endif
