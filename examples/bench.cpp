/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;
  //Geometry gg("tC0_tA0_tB0_colMaj1_m1024_n48000_k2816_lda1024_ldb2816_ldc1024_ws0_f32");
  //HyPas    hypas("A_MIC8_PAD2_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW1_MIW1_WOS0__C_UNR16_GAL2_PUN0_ICE1_IWI1_SZT1_NAW16_UFO0_MAC256_SKW10");

  // A freeze on compile case
  Geometry gg("tC0_tA0_tB0_colMaj1_m512_n8_k50000_lda512_ldb50000_ldc512_ws0_f32");
  //HyPas hypas("A_MIC6_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC3_PAD2_PLU1_LIW1_MIW1_WOS0_VEW1__C_UNR32_GAL3_PUN1_ICE8_IWI0_SZT0_NAW16_UFO0_MAC16_SKW8");
  HyPas hypas("A_MIC6_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC3_PAD2_PLU1_LIW1_MIW1_WOS0_VEW1__C_UNR32_GAL3_PUN1_ICE8_IWI0_SZT0_NAW16_UFO0_MAC16_SKW8");

///home/james/test15/gg_tC0_tA0_tB0_colMaj1_m1024_n48000_k2816_lda1024_ldb2816_ldc1024_ws0_f32/cns_/log.txt
//Profiling is not available
//A_MIC8_PAD2_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW1_MIW1_WOS0__C_UNR16_GAL2_PUN0_ICE1_IWI1_SZT1_NAW16_UFO0_MAC256_SKW10   :   6225.4 gflops 

  
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.benchgemm({hypas}, {{0, 2000}, {0., 100.}});
  return 0;
}




