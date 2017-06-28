/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
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
