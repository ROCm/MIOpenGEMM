/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

int main()
{
  using namespace MIOpenGEMM;

   
  Geometry gg("tC0_tA1_tB0_colMaj0_m2048_n121_k1_lda2048_ldb121_ldc121_ws0_f32");  
  HyPas    hp("A_MIC1_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__C_UNR32_GAL2_PUN1_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");  
    
  CLHint         devhint({"AMD", "gfx"});
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
    
  // UNR 32 -> 16 fixes. 
  //HyPas    hp = boa.find(get_at_least_n_seconds(0.00001), std::string("")).hypas;
  
  boa.accuracy_test(hp);
  mowri << "\ndone.\n";
  return 0;
}
