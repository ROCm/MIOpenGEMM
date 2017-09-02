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



  Geometry       gg("tC0_tA1_tB0_colMaj0_m4864_n32768_k19_lda4864_ldb32768_ldc32768_ws0_f32");
  CLHint         devhint({"Advanced Micro Devices", "gfx803"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo       boa(gg, offsets, mowri, devhint);

  auto find_params = get_at_least_n_restarts(6);
  Constraints constraints("");//A_WOS0__B_WOS0__C_ICE1");
    
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
  std::cout <<  soln.get_cache_entry_string();
  std::cout << " -- snip -- -- -- snip --\n\n";


  return 0;
}
