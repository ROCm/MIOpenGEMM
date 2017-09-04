/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>

int main()
{
  using namespace MIOpenGEMM;

  Geometry gg("tC0_tA0_tB0_colMaj1_m130305_n1_k1600_lda130305_ldb1600_ldc130305_ws1_f32");
  HyPas    hp({"MIC1_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1",
            "MIC1_PAD2_PLU1_LIW0_MIW1_WOS0_VEW1",
            "UNR64_GAL2_PUN1_ICE2_IWI0_SZT0_MAD0_NAW16_UFO1_MAC64_SKW7_AFI0_MIA0"});

  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  boa.accuracy_test(hp);
  mowri << "\ndone.\n";
  return 0;
}
