/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>
#include <miopengemm/redirection.hpp>


int main()
{
  using namespace MIOpenGEMM;


  Geometry gg("tC0_tA0_tB1_colMaj1_m5100_n5100_k5100_lda5100_ldb5100_ldc5100_ws100000000_f32");  
  HyPas    hp("A_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2__B_MIC8_PAD1_PLU0_LIW0_MIW0_WOS0_VEW2__C_UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0");  

  //std::cout << "\n" << gg.get_string(); 
  //std::cout << "\n" << hp.get_string() << '\n';
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo       boa(gg, offsets, mowri, devhint);
  for  (unsigned i = 0 ; i < 1; ++i){
    //std::cout << "\n\n\n";
    boa.benchgemm({hp}, {{0, 5}, {0., 100.}});
  }

  return 0;
}




