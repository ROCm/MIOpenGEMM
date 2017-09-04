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

  // this is because of ROCm 1.6 testing. TODO: raise issue.
  // Also : mention that there might be an error when k is small.

  Geometry gg("tC0_tA0_tB0_colMaj1_m32004_n1_k1728_lda32004_ldb1728_ldc32004_ws0_f32");

  CLHint         devhint({"Advanced Micro Devices", "gfx803"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  auto        find_params = get_at_least_n_restarts(1);
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
