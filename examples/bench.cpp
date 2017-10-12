/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/redirection.hpp>
#include <miopengemm/tinytwo.hpp>








int main()
{
  using namespace MIOpenGEMM;
Geometry gg{"tC0_tA0_tB1_colMaj1_m5100_n5100_k5100_lda5100_ldb5100_ldc5100_ws0_f32"}; // gg

HyPas hp{{{ //hp

"MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_LOM1_VEW2",
"MIC8_PAD1_PLU0_LIW0_MIW0_WOS0_LOM1_VEW2",
"UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0_PAK1"

}}};
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMWITHDEPS, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  for (unsigned i = 0; i < 1; ++i)
  {
    boa.benchgemm({hp}, {{{0, 4}}, {{0., 4.}}});
  }

  return 0;
}
