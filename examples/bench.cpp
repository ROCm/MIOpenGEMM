/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/redirection.hpp>


int main()
{
  using namespace MIOpenGEMM;


  Geometry gg("tC0_tA0_tB0_colMaj1_m77_n1002_k77_lda77_ldb77_ldc77_ws0_f32");  
  HyPas    hp("A_MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1__B_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");  
  //HyPas    hp("A_MIC2_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1__B_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");
  //HyPas    hp("A_MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1__B_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");  



  std::cout << "\n" << gg.get_string(); 
  std::cout << "\n" << hp.get_string() << '\n';
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  for  (unsigned i = 0 ; i < 1; ++i){
    std::cout << "\n\n\n";
    boa.benchgemm({hp}, {{0, 30}, {0., 100.}});
  }

  return 0;
}




