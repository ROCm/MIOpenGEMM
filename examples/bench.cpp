/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/redirection.hpp>
#include <miopengemm/tinytwo.hpp>







int main()
{
  using namespace MIOpenGEMM;
Geometry gg{"tC0_tA1_tB0_colMaj1_m1024_n8_k500000_lda500000_ldb500000_ldc1024_ws0_f32"}; // gg

HyPas hp{{{ //hp

"MIC1_PAD1_PLU0_LIW0_MIW0_WOS0_LOM1_VEW1",
"MIC1_PAD1_PLU1_LIW1_MIW0_WOS0_LOM1_VEW1",
"UNR128_GAL1_PUN0_ICE4_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW9_AFI0_MIA0"}}};
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMWITHDEPS, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  for (unsigned i = 0; i < 1; ++i)
  {
    boa.benchgemm({hp}, {{{0, 100}}, {{0., 100.}}});
  }

  return 0;
}
