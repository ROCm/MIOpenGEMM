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

#include <miopengemm/iterexperiments.hpp>

/* Run the 77 DeepBench problems */
int main()
{

  auto                    geometries          = MIOpenGEMM::get_deepbench_geometries();
  float                   allotted_time       = 360.00;
  unsigned                allotted_iterations = 40;
  unsigned                n_runs_per_kernel   = 3;
  MIOpenGEMM::SummaryStat sumstat(MIOpenGEMM::Max);

  bool                     verbose       = false;
  std::vector<std::string> v_constraints = {"A_WOS0__B_WOS0"};
  // set the directory to write search histories to, or set to the empty string if not needed
  std::string basedir("/home/james/miogout/overnight_16_june");

  MIOpenGEMM::FindParams find_params(
    allotted_time, allotted_iterations, n_runs_per_kernel, sumstat);

  bool        verbose_outer = true;
  std::string fn_outer("");
  MIOpenGEMM::run_find_experiments(
    geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);

  return 0;
}
