/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>

int main()
{

  using namespace MIOpenGEMM;

  bool test_accuracy_of_soln = false;
  bool bench_the_soln        = true;

  Geometry       gg("tC0_tA0_tB0_colMaj0_m1024_n1024_k1024_lda1024_ldb1024_ldc1024_ws0_f32");
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo       boa(gg, offsets, mowri, devhint);

  auto find_params = get_at_least_n_restarts(1);
  Constraints constraints("");//A_WOS0__B_WOS0__C_ICE1");
    
  Solution soln = boa.find(find_params, constraints);

  if (test_accuracy_of_soln)
  {
    mowri << "\n\n\nAccuracy\n";
    boa.accuracy_test(soln.hypas);
  }

  if (bench_the_soln)
  {
    mowri << "\n\n\nBenchmark\n";
    boa.benchgemm({soln.hypas}, {{0, 11}, {0, 1000.}});
  }

  return 0;
}
