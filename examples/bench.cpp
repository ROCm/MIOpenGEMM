/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>
#include <miopengemm/redirection.hpp>


int main()
{
  using namespace MIOpenGEMM;


  Geometry gg("tC0_tA0_tB1_colMaj1_m1555_n1555_k1555_lda1555_ldb1555_ldc1555_ws100000000_f32");  
  HyPas    hp("A_MIC4_PAD1_PLU0_LIW0_MIW1_WOS2_VEW1__B_MIC8_PAD1_PLU0_LIW0_MIW0_WOS2_VEW2__C_UNR16_GAL3_PUN0_ICE7_IWI1_SZT0_NAW16_UFO0_MAC256_SKW10_AFI1_MIA0_MAD1");  

  //std::cout << "\n" << gg.get_string(); 
  //std::cout << "\n" << hp.get_string() << '\n';
  CLHint  devhint;
  Offsets  offsets = get_zero_offsets();
  owrite::Writer  mowri(Ver::E::TERMWITHDEPS, "");
  dev::TinyTwo      boa(gg, offsets, mowri, devhint);

  for  (unsigned i = 0 ; i < 1; ++i){
    //std::cout << "\n\n\n";
    boa.benchgemm({hp}, {{0, 5}, {0., 100.}});
  }

  return 0;
}




