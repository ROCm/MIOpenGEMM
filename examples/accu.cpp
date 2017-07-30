/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;

   
  Geometry gg("tC0_tA0_tB0_colMaj1_m512_n512_k128_lda512_ldb5000_ldc512_ws0_f64");
  HyPas hypas("A_MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1__C_UNR8_GAL2_PUN1_ICE1_IWI1_SZT0_NAW16_UFO1_MAC64_SKW10_AFI1_MIA0");
               
  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}
