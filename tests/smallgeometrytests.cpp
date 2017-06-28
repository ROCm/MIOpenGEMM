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
#include <miopengemm/basicfind.hpp>
#include <miopengemm/miogemm.hpp>

template <typename TFloat>
void geometrytest(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k)
{

  unsigned lda = (tA == isColMajor ? k : m) + 1;
  unsigned ldb = (tB == isColMajor ? n : k) + 2;
  unsigned ldc = (tC == isColMajor ? n : m) + 4;

  unsigned a_offset = 5;
  unsigned b_offset = 7;
  unsigned c_offset = 11;

  unsigned tail_off_a = 13;
  unsigned tail_off_b = 17;
  unsigned tail_off_c = 19;

  float                   allotted_time       = 0.001;
  unsigned                allotted_iterations = 1;
  unsigned                n_runs_per_kernel   = 1;
  MIOpenGEMM::SummaryStat sumstat             = MIOpenGEMM::Median;

  // set verbose to true if you want output to terminal
  bool verbose = false;
  // set logfile if you want output forked to file
  std::string logfile("");

  std::string constraints_string = "A_WOS0__B_WOS0__C_ICE3";

  unsigned n_postfind_runs = 1;
  bool     do_cpu_test     = true;

  unsigned             workspace_size   = 3;
  unsigned             workspace_offset = 4;
  char                 floattype        = sizeof(TFloat) == 4 ? 'f' : 'd';
  MIOpenGEMM::Geometry gg(
    isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
  MIOpenGEMM::Offsets offsets(
    a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c);
  MIOpenGEMM::FindParams find_params(
    allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);
  MIOpenGEMM::basicfind(
    gg, offsets, find_params, verbose, logfile, constraints_string, n_postfind_runs, do_cpu_test);
}

int main()
{
  unsigned m     = 55;
  unsigned k     = 118;
  unsigned testi = 0;
  for (bool tC : {false, true})
  {
    for (bool isColMajor : {false, true})
    {
      for (bool tA : {false, true})
      {
        for (bool tB : {false, true})
        {
          for (unsigned n : {m - 10, m + 10})
          {
            testi += 1;
            k += 1;
            std::cout << "\ntest " << testi << "/32";
            std::cout << "\nm=" << m << " n=" << n << " k=" << k << "\ntA=" << tA << " tB=" << tB
                      << " tC=" << tC << " isColMajor=" << isColMajor << std::endl;
            std::cout << "<float>  ";
            geometrytest<float>(isColMajor, tA, tB, tC, m, n, k);
            std::cout << "<double> ";
            geometrytest<double>(isColMajor, tA, tB, tC, m, n, k);
          }
        }
      }
    }
  }
  return 0;
}
