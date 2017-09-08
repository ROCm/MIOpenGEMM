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


  //Geometry gg("tC0_tA0_tB1_colMaj1_m3275_n4860_k64_lda3275_ldb4864_ldc3275_ws0_f32");
  //Geometry gg("tC0_tA1_tB0_colMaj1_m3072_n48000_k1024_lda1024_ldb1024_ldc3072_ws52594944_f32");


  Geometry gg("tC0_tA1_tB0_colMaj1_m307_n4800_k1024_lda1024_ldb1024_ldc3072_ws0_f32");
  
  
  //6249.987
  
  //tC=0 tA=1 tB=0 colMaj=1 m=3072   n=48000  k=1024   lda=1024   ldb=1024   ldc=3072   ws=52594944 f=32  time[ms]:54.058416   gflops:5586.36


  CLHint         devhint({"Advanced Micro Devices", "gfx803"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  auto        find_params = get_at_least_n_restarts(2);
  Constraints constraints("");

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
