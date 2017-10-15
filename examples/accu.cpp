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


  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets(gg.wSpaceSize.size());  // get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  boa.accuracy_test(hp);
  mowri << "\ndone.\n";
  return 0;
}
