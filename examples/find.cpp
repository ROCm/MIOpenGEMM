/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
//#include <miopengemm/findparams.hpp>

int main()
{

  using namespace MIOpenGEMM;

  bool test_accuracy_of_soln = false;
  bool bench_the_soln        = false;



  Geometry       gg("tC0_tA1_tB0_colMaj0_m363_n45796_k1_lda363_ldb45796_ldc45796_ws0_f32");
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);

  auto find_params = get_at_least_n_restarts(10);
  Constraints constraints("");
    
  Solution soln = boa.find(find_params, constraints);

  if (test_accuracy_of_soln)
  {
    mowri << "\n\n\nAccuracy\n";
    boa.accuracy_test(soln.hypas);
  }

  if (bench_the_soln)
  {
    mowri << "\n\n\nBenchmark\n";
    boa.benchgemm({soln.hypas}, {{0, 11}, {0, 100.}});
  }

  return 0;
}
