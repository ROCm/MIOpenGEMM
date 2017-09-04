/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry    gg("tC0_tA0_tB0_colMaj0_m800_n1200_k1000_lda2000_ldb2000_ldc2000_ws0_f32");
  Constraints constraints("C_ICE1");
  // To search for at least n=20 seconds we use this factory function
  FindParams     find_params = get_at_least_n_seconds(3);
  Offsets        offsets     = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TRACK, "");  // TERMINAL
  CLHint         devhint;
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);
  Solution       soln = boa.find2(find_params, constraints);
  std::cout << " \n\n\nThe following string can be cut and paste into kernelcache.cpp";
  std::cout << " \n\n-- snip -- -- -- snip --\n\n";
  std::cout << soln.get_cache_entry_string();
  std::cout << " -- snip -- -- -- snip --\n\n";
  return 0;
}
