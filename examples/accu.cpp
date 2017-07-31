/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;

   
  Geometry gg("tC0_tA0_tB1_colMaj1_m125_n25_k255_lda130_ldb25_ldc1250_ws0_f32");  
  HyPas    hp("A_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2__B_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL3_PUN1_ICE1_IWI0_SZT0_NAW16_UFO1_MAC4_SKW10_AFI1_MIA0");  
               
  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hp);
  mowri << "\ndone.\n";
  return 0;
}
