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


  //gives incorrect results
  Geometry gg_0("tC0_tA1_tB0_colMaj0_m2048_n121_k1_lda2048_ldb121_ldc121_ws0_f32");  
  HyPas    hp_0("A_MIC1_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__C_UNR32_GAL2_PUN1_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");  


  //gives incorrect results
  Geometry gg_1("tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws1_f32");
  Hypas    hp_1("A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9"); 
 
  // A freeze on compile case
  Geometry gg_2("tC0_tA0_tB0_colMaj1_m2560_n65_k2560_lda2560_ldb2560_ldc2560_ws0_f32");
  HyPas    hp_2("A_MIC6_PAD2_PLU1_LIW0_MIW0_WOS0__B_MIC6_PAD1_PLU0_LIW1_MIW0_WOS0__C_UNR16_GAL3_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC16_SKW8");

  // A freeze on compile case: 
  Geometry gg_4("tC0_tA0_tB0_colMaj1_m512_n8_k500000_lda512_ldb500000_ldc512_ws0_f32");
  HyPas    hp_4("A_MIC6_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC3_PAD2_PLU1_LIW1_MIW1_WOS0_VEW1__C_UNR32_GAL3_PUN1_ICE8_IWI0_SZT0_NAW16_UFO0_MAC16_SKW8");

  // A freeze on compile case: 
  Geometry gg_5("tC0_tA0_tB0_colMaj1_m512_n8_k500000_lda512_ldb500000_ldc512_ws0_f32");
  HyPas    hp_5("A_MIC6_PAD2_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC6_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL2_PUN1_ICE2_IWI1_SZT0_NAW16_UFO1_MAC4_SKW9");

  // A freeze on compile case: 
  Geometry gg_6("tC0_tA1_tB0_colMaj1_m363_n1_k576_lda576_ldb576_ldc363_ws0_f32");
  HyPas    hp_6("A_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD2_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL2_PUN1_ICE1_IWI0_SZT0_NAW64_UFO0_MAC64_SKW7");

  // A freeze on compile case: 
  Geometry gg_7("tC0_tA0_tB0_colMaj1_m25_n5_k25_lda25_ldb25_ldc25_ws0_f32");
  HyPas    hp_7("A_MIC1_PAD0_PLU1_LIW1_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW1_MIW0_WOS0_VEW1__C_UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC4_SKW11_AFI1_MIA0");

  // A freeze on compile case: 
  Geometry gg_8("tC0_tA0_tB0_colMaj1_m77_n1002_k77_lda77_ldb77_ldc77_ws0_f32");  
  HyPas    hp_8("A_MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1__B_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");  
  
  // TODO : the rocm filter:
  // they all have minimal SKW. 
  
  CLHint         devhint;
  Offsets        offsets = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}




