/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>

// test if (Geometry, HyPas) gives correct results
int main()
{
  using namespace MIOpenGEMM;

  // Geometry gg{"tC0_tA0_tB0_colMaj1_m25_n5_k25_lda25_ldb25_ldc25_ws0_f32"};

  // Geometry gg{"tC0_tA0_tB0_colMaj1_m25_n5_k25_lda25_ldb25_ldc25_ws0_f32"};

  // Geometry gg = MIOpenGEMM::get_squareNN_geometry<float>(1023);
  Geometry gg{"tC0_tA0_tB1_colMaj1_m1023_n1023_k1023_lda1023_ldb1023_ldc1023_ws1_f32"};

  HyPas hp({"MIC8_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1",
            "MIC4_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1",
            "UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0"});

  //  HyPas hp({"MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2", "MIC4_PAD2_PLU0_LIW0_MIW1_WOS0_VEW2",
  //  "UNR8_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0"});

  // HyPas hp{{"MIC1_PAD0_PLU1_LIW1_MIW1_WOS0_VEW1",
  //"MIC1_PAD0_PLU1_LIW1_MIW0_WOS0_VEW1",
  //"UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC4_SKW11_AFI1_MIA0_MAD0"}};

  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  boa.accuracy_test(hp);
  mowri << "\ndone.\n";
  return 0;
}
