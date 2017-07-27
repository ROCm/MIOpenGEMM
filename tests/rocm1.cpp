/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;

  /* crashes if all ints are unsigned in OpenCL kernel. Changing certain to ushort stops crash */ 
  HyPas hypas({
   "MIC6_PAD1_PLU1_LIW0_MIW0_WOS0", 
   "MIC5_PAD1_PLU1_LIW0_MIW1_WOS0", 
   "UNR8_GAL3_PUN1_ICE1_NAW16_UFO0_MAC4_SKW9"});  
  Geometry gg("tC0_tA0_tB0_colMaj1_m1024_n8_k10000_lda1024_ldb10000_ldc1024_ws1_f32");


  // A freeze on compile case
  Geometry gg_2("tC0_tA0_tB0_colMaj1_m2560_n65_k2560_lda2560_ldb2560_ldc2560_ws0_f32");
  HyPas    hypas_2("A_MIC6_PAD2_PLU1_LIW0_MIW0_WOS0__B_MIC6_PAD1_PLU0_LIW1_MIW0_WOS0__C_UNR16_GAL3_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC16_SKW8");

  // A freeze on compile case: 
  Geometry gg_4("tC0_tA0_tB0_colMaj1_m512_n8_k500000_lda512_ldb500000_ldc512_ws0_f32");
  HyPas hypas_4("A_MIC6_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC3_PAD2_PLU1_LIW1_MIW1_WOS0_VEW1__C_UNR32_GAL3_PUN1_ICE8_IWI0_SZT0_NAW16_UFO0_MAC16_SKW8");


  // above : is it perhaps  MAC16 with SKW8 ?

  //gives incorrect results
  Hypas hp3("A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9");
  Geometry gg3("tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws1_f32");

  Geometry gg4("tC0_tA0_tB0_colMaj1_m512_n8_k500000_lda512_ldb500000_ldc512_ws0_f32");
  HyPas hp4("A_MIC6_PAD2_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC6_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL2_PUN1_ICE2_IWI1_SZT0_NAW16_UFO1_MAC4_SKW9");


  CLHint         devhint;
  Offsets        offsets = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}




