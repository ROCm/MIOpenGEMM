/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry    gg("tC0_tA0_tB0_colMaj0_m28_n2048_k2048_lda2048_ldb2048_ldc2048_ws5000000_f32");
  Constraints constraints("A_MIC1_PAD1_PLU0_LIW0_MIW1_WOS2__C_ICE3");
  // To search for at least n=20 seconds we use this factory function
  FindParams     find_params = get_at_least_n_seconds(20);
  Offsets        offsets     = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TRACK, "");
  CLHint         devhint;
  dev::Boa       boa(gg, offsets, mowri, devhint);
  Solution       soln = boa.find(find_params, constraints);
  std::cout << " \n\n\nThe following string can be cut and paste into kernelcache.cpp";
  std::cout << " \n\n-- snip -- -- -- snip --\n\n";
  std::cout << soln.get_cache_entry_string();
  std::cout << " -- snip -- -- -- snip --\n\n";
  return 0;
}
