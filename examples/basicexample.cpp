/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <miopengemm/basicfind.hpp>
#include <miopengemm/miogemm.hpp>

template <typename TFloat>
void basicexample()
{
  // define the GEMM problem
  bool isColMajor = true;
  bool tA         = true;
  bool tB         = false;
  bool tC         = false;

  size_t m = 1000;
  size_t n = 128;
  size_t k = 4096;

  size_t lda = (tA == isColMajor ? k : m) + 0;
  size_t ldb = (tB == isColMajor ? n : k) + 0;
  size_t ldc = (tC == isColMajor ? n : m) + 0;

  size_t a_offset = 1;
  size_t b_offset = 2;
  size_t c_offset = 3;

  // tails are not nec, but help in debugging
  size_t tail_off_a = 13;
  size_t tail_off_b = 19;
  size_t tail_off_c = 33;

  // define how long to search for, in seconds.
  // No kernels will be compiled after this allotted time
  float                   allotted_time     = 1.01;
  size_t                allotted_descents = 1;
  size_t                n_runs_per_kernel = 3;
  MIOpenGEMM::SummStat::E sumstat           = MIOpenGEMM::SummStat::E::MAX;

  MIOpenGEMM::FindParams find_params(allotted_time, allotted_descents, n_runs_per_kernel, sumstat);

  // print output to terminal (true) or complete silence to terminal (false)
  bool verbose = true;
  bool use_mowri_tracker = false;
  // print output to logfile (non-empty string) or not (empty string)
  // MUST BE SET BY USER
  std::string logfile("basicexample-findlog.txt");
  // enforce that the kernel is deterministic, or not. Note that
  // for small problems, non-deterministic kernels are significantly (2x) faster
  std::string constraints_string = "";
  size_t    n_postfind_runs    = 5;
  bool        do_cpu_test        = true;

  size_t workspace_size   = 3;
  size_t workspace_offset = 4;
  char     floattype        = 'f';

  MIOpenGEMM::Geometry gg(
    isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
  MIOpenGEMM::Offsets offsets(
    a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c);
  
  
  MIOpenGEMM::basicfind(
    gg, offsets, find_params, verbose, logfile, constraints_string, n_postfind_runs, do_cpu_test, use_mowri_tracker);
}

int main()
{
  basicexample<float>(); /* or example<double> for dgemm example */
  return 0;
}
