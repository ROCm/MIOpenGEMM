/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA0_tB0_colMaj1_m1024_n8_k500000_lda1024_ldb500000_ldc1024_ws0_f32");
  HyPas    hypas(  // hp
    "A_MIC8_PAD1_PLU1_LIW0_MIW0_WOS0__B_MIC6_PAD2_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_ICE4_NAW64_"
    "UFO0_MAC4_SKW9");

  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.benchgemm({hypas}, {{0, 3}, {0., 100.}});
  return 0;
}
