/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/redirection.hpp>


int main()
{
  using namespace MIOpenGEMM;

  // A freeze on compile case
  Geometry gg("tC0_tA0_tB0_colMaj1_m25_n5_k25_lda25_ldb25_ldc25_ws0_f32");
  //HyPas hypas("A_MIC1_PAD0_PLU1_LIW1_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW1_MIW0_WOS0_VEW1__C_UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC4_SKW11_AFI1_MIA0");
  
  
  HyPas hypas("A_MIC1_PAD0_PLU1_LIW1_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW1_MIW0_WOS0_VEW1__C_UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC4_SKW11_AFI1_MIA0");
  
  std::cout << "\n" << gg.get_string(); 
  std::cout << "\n" << hypas.get_string() << '\n';
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  for  (unsigned i = 0 ; i < 1; ++i){
    std::cout << "\n\n\n";
    boa.benchgemm({hypas}, {{0, 30}, {0., 100.}});
  }

  return 0;
}




