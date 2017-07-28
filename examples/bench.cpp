/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;

  // A freeze on compile case


  Geometry gg("tC0_tA0_tB1_colMaj1_m3600_n3600_k3600_lda3600_ldb3600_ldc3600_ws0_f32");
  

HyPas hypas({"MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2",
"MIC8_PAD1_PLU0_LIW0_MIW0_WOS0_VEW2",
"UNR16_GAL3_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC256_SKW10"});

  
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  boa.benchgemm({hypas}, {{0, 200000}, {0., 100.}});
  return 0;
}




