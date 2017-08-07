/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;

  //Geometry gg("tC1_tA1_tB1_colMaj1_m45_n56_k64_lda64_ldb64_ldc64_ws1_f32");    
  Geometry gg = get_squareNN_geometry<float>(513);

  HyPas    hp("A_MIC4_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR8_GAL3_PUN1_ICE1_IWI0_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");     
  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
      
  boa.accuracy_test(hp);
  mowri << "\ndone.\n";
  return 0;
}
