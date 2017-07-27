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

  Geometry       gg("tC0_tA0_tB0_colMaj1_m1024_n32_k512_lda1024_ldb512_ldc1024_ws0_f32");
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);

  // FindParams find_params = get_quick_find_params();
  auto find_params = get_at_least_n_seconds(10);  //(100, 1000., 3, 1., SummStat::E::MAX);
  Constraints constraints("");//A_MIC8_PAD1_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC6_PAD1_PLU0_LIW0_MIW0_WOS0_VEW1__C_UNR16_GAL2_PUN0_ICE1_IWI0_SZT0_NAW64_UFO0_MAC64_SKW10");
  

//  A_MIW0_VEW1__B_MIW0_VEW1");
  
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
