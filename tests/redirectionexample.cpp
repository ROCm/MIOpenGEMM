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
#include <string>
#include <miopengemm/redirection.hpp>

struct ExampleGeometry
{
  bool        isColMajor, tA, tB, tC;
  unsigned    m, n;
  std::string a, b;
  ExampleGeometry(bool        isColMajor_,
                  bool        tA_,
                  bool        tB_,
                  bool        tC_,
                  unsigned    m_,
                  unsigned    n_,
                  std::string a_,
                  std::string b_)
    : isColMajor(isColMajor_), tA(tA_), tB(tB_), tC(tC_), m(m_), n(n_), a(a_), b(b_)
  {
  }
  ExampleGeometry(const ExampleGeometry& eg) = default;
};

// Convert to RowMajor, not TT, m > n
int main()
{

  ExampleGeometry g_in(true, true, true, false, 10, 50, "a", "b");
  ExampleGeometry g_out(g_in);
  MIOpenGEMM::redirection::redirect(
    g_out.isColMajor, g_out.tA, g_out.tB, g_out.tC, g_out.m, g_out.n, g_out.a, g_out.b);

  std::cout << "=========================\n";
  std::cout << "Canonical form conversion\n";
  std::cout << "=========================\n"
            << "colMaj : " << g_in.isColMajor << " --> " << g_out.isColMajor << "\n"
            << "tA     : " << g_in.tA << " --> " << g_out.tA << "\n"
            << "tB     : " << g_in.tB << " --> " << g_out.tB << "\n"
            << "tC     : " << g_in.tC << " --> " << g_out.tC << "\n"
            << "m      : " << g_in.m << " --> " << g_out.m << "\n"
            << "n      : " << g_in.n << " --> " << g_out.n << "\n"
            << "a      : " << g_in.a << " --> " << g_out.a << "\n"
            << "b      : " << g_in.b << " --> " << g_out.b << "\n";
  std::cout << "=========================\n";
}
