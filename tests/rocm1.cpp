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

  // gives incorrect results. Unresolved.
  // hyperstring =
  // "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9";
  //"tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws1_f32"

  // hang before single descent rocm 1.6 night run
  // tC0_tA1_tB0_colMaj1_m512_n48000_k2816_lda2816_ldb2816_ldc512_ws1_f32


  CLHint         devhint;
  Offsets        offsets = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}




