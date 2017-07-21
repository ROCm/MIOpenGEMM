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

  Geometry       gg("tC0_tA0_tB0_colMaj1_m400_n400_k400_lda8448_ldb2816_ldc8448_ws0_f32");
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);

  // FindParams find_params = get_quick_find_params();
  auto find_params = get_at_least_n_seconds(1003.14);  //(100, 1000., 3, 1., SummStat::E::MAX);
  Constraints constraints(
    "");  // A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC5_PAD1_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL2_PUN1_ICE1_NAW16_UFO0_MAC256_SKW10");
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
