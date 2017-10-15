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

  // Geometry gg("tC0_tA0_tB1_colMaj1_m3275_n4860_k64_lda3275_ldb4864_ldc3275_ws0_f32");
  // Geometry gg("tC0_tA1_tB0_colMaj1_m3072_n48000_k1024_lda1024_ldb1024_ldc3072_ws52594944_f32");
  // Geometry gg("tC0_tA0_tB0_colMaj0_m13_n13_k13_lda13_ldb13_ldc13_ws0_f32");

  // gg{"tC0_tA1_tB0_colMaj1_m1024_n8_k500000_lda500000_ldb500000_ldc1024_ws0_f32"}; //

  Geometry      gg  = MIOpenGEMM::get_squareNN_geometry<float>(500);
  CLHint         devhint({"Advanced Micro Devices", "gfx803"});
  Offsets        offsets = get_zero_offsets(0); // no workspaces
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);
  auto           find_params = get_at_least_n_restarts(10);
  
  
  Constraints    constraints{{{
  
"MIC4_PAD1_PLU1_LIW0_MIW0_WOS0_VEW1_LOM1", 
"MIC4_PAD2_PLU1_LIW0_MIW0_WOS0_VEW1_LOM1", 
"UNR64_GAL3_PUN1_ICE1_IWI0_SZT0_MAD0_NAW16_UFO0_MAC64_SKW10_AFI0_MIA0_PAK0"

  
  }}}; 
  

  
  //A_LOM1__B_LOM1");//A_LOM0__B_LOM0");
  Solution       soln = boa.find2(find_params, constraints);
  if (test_accuracy_of_soln)
  {
    mowri << "\n\n\nAccuracy\n";
    boa.accuracy_test(soln.hypas);
  }

  if (bench_the_soln)
  {
    mowri << "\n\n\nBenchmark\n";
    boa.benchgemm({{soln.hypas}}, {{{0, 11}}, {{0, 1000.}}});
  }

  std::cout << " \n\n-- snip -- -- -- snip --\n\n";
  std::cout << soln.get_cache_entry_string();
  std::cout << " -- snip -- -- -- snip --\n\n";

  return 0;
}
