/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA0_tB0_colMaj0_m28_n2048_k2048_lda2048_ldb2048_ldc2048_ws5000000_f32");
  HyPas    hypas({"MIC1_PAD1_PLU0_LIW0_MIW1_WOS2",
               "MIC8_PAD1_PLU0_LIW0_MIW1_WOS2",
               "UNR8_GAL3_PUN1_ICE1_NAW64_UFO1_MAC64_SKW10"});

  CLHint         devhint;
  Offsets        offsets = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.accuracy_test(hypas);
  mowri << "\ndone.\n";
  return 0;
}
