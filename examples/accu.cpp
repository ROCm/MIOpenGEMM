/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>

// test if (Geometry, HyPas) gives correct results
int main()
{
  using namespace MIOpenGEMM;


  Geometry gg{"tC0_tA0_tB1_colMaj1_m6004_n6004_k6004_lda6004_ldb6004_ldc6004_ws500_ws50000001_ws50000000_f32"};

  HyPas hp{{{ //hp
  
  "MIC8_PAD1_PLU0_LIW0_MIW0_WOS1_LOM1_VEW1",
  "MIC6_PAD1_PLU0_LIW0_MIW1_WOS1_LOM1_VEW1",
  "UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI0_MIA0_PAK0"}}};

<<<<<<< HEAD
  // Geometry gg{"tC0_tA1_tB0_colMaj0_m27_n50176_k1_lda27_ldb50176_ldc50176_ws0_f32"};
  // HyPas hp = {{{"MIC2_PAD1_PLU0_LIW1_MIW1_WOS0_VEW1", "MIC4_PAD1_PLU0_LIW0_MIW0_WOS0_VEW4",
  // "UNR32_GAL1_PUN0_ICE1_IWI1_SZT0_MAD1_NAW16_UFO0_MAC256_SKW11_AFI1_MIA1"}}};

  // Geometry gg{"tC0_tA0_tB1_colMaj1_m47524_n363_k1_lda47524_ldb363_ldc47524_ws0_f32"};

  //Geometry gg{"tC0_tA0_tB0_colMaj0_m4_n400_k200_lda200_ldb400_ldc400_ws0_f32"};


//kc.add(
//{"gfx803",  // dev
//{""},  // con
//{"tC0_tA0_tB0_colMaj1_m363_n1_k363_lda363_ldb363_ldc363_ws0_f32"}}, // gg
//{{{ //hp
//}}});

  Geometry gg{"tC0_tA0_tB0_colMaj1_m363_n1_k363_lda363_ldb363_ldc363_ws0_f32"};

  HyPas hp = {{{// hp

"MIC3_PAD1_PLU1_LIW1_MIW0_WOS0_VEW1",
"MIC1_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1",
"UNR2_GAL1_PUN1_ICE6_IWI1_SZT0_MAD0_NAW64_UFO0_MAC1_SKW10_AFI1_MIA0"
}}};

//"MIC1_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1", "MIC3_PAD1_PLU1_LIW1_MIW0_WOS0_VEW1", "UNR2_GAL2_PUN1_ICE6_IWI1_SZT0_MAD0_NAW64_UFO0_MAC1_SKW10_AFI0_MIA1"}}};

  //"MIC4_PAD1_PLU0_LIW0_MIW0_WOS0_VEW4",
  //"MIC2_PAD1_PLU0_LIW1_MIW1_WOS0_VEW1",
  //"UNR32_GAL2_PUN0_ICE1_IWI1_SZT0_MAD1_NAW16_UFO0_MAC256_SKW9_AFI0_MIA0"}}};

  //  HyPas hp({"MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2", "MIC4_PAD2_PLU0_LIW0_MIW1_WOS0_VEW2",
  //  "UNR8_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0"});

  // HyPas hp{{"MIC1_PAD0_PLU1_LIW1_MIW1_WOS0_VEW1",
  //"MIC1_PAD0_PLU1_LIW1_MIW0_WOS0_VEW1",
  //"UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC4_SKW11_AFI1_MIA0_MAD0"}};
=======
>>>>>>> develop

  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets(gg.wSpaceSize.size());  // get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  boa.accuracy_test(hp);
  mowri << "\ndone.\n";
  return 0;
}
