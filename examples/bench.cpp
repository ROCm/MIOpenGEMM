/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/redirection.hpp>


int main()
{
  using namespace MIOpenGEMM;


  Geometry gg("tC0_tA1_tB0_colMaj1_m1024_n16_k500000_lda500000_ldb500000_ldc1024_ws0_f32");  
  HyPas    hp("A_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE11_IWI1_SZT0_NAW64_UFO0_MAC256_SKW10_AFI0_MIA1");  


//(14 / 160)tC0_tA1_tB0_colMaj1_m1024_n16_k500000_lda500000_ldb500000_ldc1024_ws0_f32
//soln1 : A_MIC4_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1__B_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE11_IWI1_SZT0_NAW64_UFO0_MAC256_SKW10_AFI1_MIA0
//soln2 : A_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE11_IWI1_SZT0_NAW64_UFO0_MAC256_SKW10_AFI0_MIA1



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




