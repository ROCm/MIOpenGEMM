/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;

  //TODO : investigate and catch why
  //Geometry gg("tC0_tA0_tB1_colMaj1_m5_n5_k133_lda6_ldb6_ldc6_ws10000000_f32");
  //HyPas    hypas(
              //{"MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1",
               //"MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1",
               //"UNR4_GAL1_PUN0_ICE1_NAW64_UFO0_MAC1_SKW10_IWI0_SZT0"});



//
   
   
  Geometry gg("tC0_tA0_tB1_colMaj1_m130_n300_k38_lda5000_ldb5000_ldc1000_ws10000000_f32");
  HyPas    hypas(
  "A_MIC8_PAD1_PLU0_LIW1_MIW1_WOS0_VEW4__B_MIC6_PAD2_PLU0_LIW0_MIW1_WOS0_VEW2__C_UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW16_UFO1_MAC256_SKW11");

              //{"MIC4_PAD0_PLU0_LIW1_MIW0_WOS0_VEW2",
               //"MIC4_PAD0_PLU0_LIW0_MIW1_WOS1_VEW4",
               //"UNR16_GAL1_PUN0_ICE3_NAW64_UFO0_MAC64_SKW11_IWI0_SZT0"});
               
  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets();//get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}
