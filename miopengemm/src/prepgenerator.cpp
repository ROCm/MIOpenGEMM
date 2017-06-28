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
#include <miopengemm/prepgenerator.hpp>

namespace MIOpenGEMM
{
namespace prepgen
{

void PrepGenerator::set_usage_from_matrixchar()
{

  uses_alpha = false;
  if (matrixchar == 'c')
  {
    uses_a         = false;
    uses_b         = false;
    uses_c         = true;
    uses_workspace = false;
    uses_beta      = true;
  }

  else
  {
    uses_c         = false;
    uses_workspace = true;
    uses_beta      = false;

    if (matrixchar == 'a')
    {
      uses_a = true;
      uses_b = false;
    }
    else if (matrixchar == 'b')
    {
      uses_a = false;
      uses_b = true;
    }
    else
    {
      throw miog_error("Unrecognised matrixchar in forallgenerator.cpp : " +
                       std::string(1, matrixchar) + std::string(".\n"));
    }
  }
}

void PrepGenerator::append_basic_what_definitions(std::stringstream& ss)
{
  ss << "#define TFLOAT  " << dp.t_float << "\n"
     << "#define LD" << MATRIXCHAR << " " << gg.ldX.at(emat_x) << "\n"
     << "/* less than or equal to LD" << MATRIXCHAR
     << ", DIM_COAL is size in the contiguous direction (m for c matrix if col "
     << "contiguous and not "
     << "transposed) */ \n"
     << "#define DIM_COAL " << gg.get_coal(emat_x) << "\n"
     << "/* DIM_UNCOAL is the other dimension of the matrix */ \n"
     << "#define DIM_UNCOAL " << gg.get_uncoal(emat_x) << "\n\n";
}

PrepGenerator::PrepGenerator(const hyperparams::HyperParams&     hp_,
                             const Geometry&                     gg_,
                             const derivedparams::DerivedParams& dp_,
                             std::string                         type_)
  : basegen::BaseGenerator(hp_, gg_, dp_, type_)
{
}
}
}
