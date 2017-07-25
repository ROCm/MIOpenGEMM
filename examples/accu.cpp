/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;
  //Geometry gg("tC0_tA0_tB1_colMaj1_m512_n512_k512_lda512_ldb512_ldc512_ws10000000_f32");
  Geometry gg("tC0_tA0_tB1_colMaj1_m4_n4_k1_lda4_ldb4_ldc4_ws10000000_f32");
  HyPas    hypas(
              {"MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW2",
               "MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW2",
               "UNR2_GAL1_PUN0_ICE1_NAW64_UFO0_MAC1_SKW10_IWI0_SZT0"});

  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets();//get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}
