/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/redirection.hpp>
#include <miopengemm/tinytwo.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA0_tB1_colMaj1_m8188_n8188_k8188_lda8188_ldb8188_ldc8188_ws0_f32");
  // Geometry gg = MIOpenGEMM::get_squareNN_geometry<float>(8188);

  HyPas hp{{{ //hp
    
    "MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2",
"MIC8_PAD1_PLU0_LIW0_MIW0_WOS0_VEW2",
"UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0"}}};

    //"MIC8_PAD1_PLU0_LIW1_MIW1_WOS0_VEW4",
    //"MIC6_PAD2_PLU0_LIW0_MIW0_WOS0_VEW1",
    //"UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW11_AFI0_MIA0"}}};


  // HyPas hp({"MIC8_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2",
  //"MIC5_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1",
  //"UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0"});

  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMWITHDEPS, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  for (unsigned i = 0; i < 1; ++i)
  {
    boa.benchgemm({hp}, {{{0, 10}}, {{0., 100.}}});
  }

  return 0;
}
