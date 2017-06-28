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

#include <string>
#include <miopengemm/floattostring.hpp>

namespace MIOpenGEMM
{
namespace floattostring
{

std::string float_string_type(double x)
{
  (void)x; /* Keeping -Wunused-parameter warning quiet */
  return "double";
}

std::string float_string_type(float x)
{
  (void)x; /* Keeping -Wunused-parameter warning quiet */
  return "float";
}

char float_char_type(double x)
{
  (void)x; /* Keeping -Wunused-parameter warning quiet */
  return 'd';
}

char float_char_type(float x)
{
  (void)x; /* Keeping -Wunused-parameter warning quiet */
  return 'f';
}

std::string get_float_string(char floattype)
{
  if (floattype == 'f')
  {
    return "float";
  }
  else
  {
    return "double";
  }
}
}
}
