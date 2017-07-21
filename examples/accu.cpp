/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA0_tB0_colMaj0_m32_n32_k155_lda512_ldb2048_ldc2048_ws5000000_f32");
  HyPas    hypas(
              {"MIC2_PAD1_PLU0_LIW0_MIW1_WOS0",
               "MIC2_PAD1_PLU0_LIW0_MIW1_WOS0",
               "UNR8_GAL1_PUN0_ICE3_NAW64_UFO0_MAC64_SKW10_IWI0"});

  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();//get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}
