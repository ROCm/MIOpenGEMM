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


  Geometry gg("tC0_tA0_tB0_colMaj1_m130305_n1_k1600_lda130305_ldb1600_ldc130305_ws0_f32");


//(jn) newly found soln hypas "MIC1_PAD0_PLU1_LIW0_MIW1_WOS0_VEW1", "MIC1_PAD0_PLU0_LIW1_MIW1_WOS0_VEW1", "UNR64_GAL3_PUN1_ICE1_IWI0_SZT1_MAD1_NAW16_UFO1_MAC64_SKW7_AFI1_MIA0"
//this is for network config : tC0_tA0_tB0_colMaj1_m130305_n1_k1600_lda130305_ldb1600_ldc130305_ws0_f32
//about to run with network config : tC0_tA0_tB0_colMaj1_m130305_n1_k1600_lda130305_ldb1600_ldc130305_ws0_f32 (jn GEMM) 3.79201  


  CLHint         devhint({"Advanced Micro Devices", "gfx803"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  auto        find_params = get_at_least_n_restarts(10);
  Constraints constraints("C_MAC64");

  Solution soln = boa.find2(find_params, constraints);

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

  std::cout << " \n\n-- snip -- -- -- snip --\n\n";
  std::cout << soln.get_cache_entry_string();
  std::cout << " -- snip -- -- -- snip --\n\n";

  return 0;
}
