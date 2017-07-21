/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA1_tB0_colMaj1_m1760_n64_k1760_lda1760_ldb1760_ldc1760_ws0_f32");
  HyPas    hypas(
{ // hp
"MIC4_PAD1_PLU1_LIW0_MIW1_WOS0",
"MIC4_PAD1_PLU0_LIW1_MIW0_WOS0",
"UNR16_GAL3_PUN0_ICE9_IWI1_NAW64_UFO0_MAC256_SKW10"}
  
  );

  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.benchgemm({hypas}, {{0, 10}, {0., 100.}});
  return 0;
}
