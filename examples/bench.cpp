/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>
#include <miopengemm/redirection.hpp>


int main()
{
  using namespace MIOpenGEMM;


  Geometry gg("tC0_tA0_tB0_colMaj1_m550_n550_k550_lda550_ldb550_ldc550_ws0_f32");  
  HyPas    hp("A_MIC8_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC8_PAD2_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL1_PUN0_ICE1_IWI0_SZT0_MAD1_NAW64_UFO0_MAC64_SKW10_AFI1_MIA1");  


//geometry : tC0_tA0_tB0_colMaj1_m550_n550_k550_lda550_ldb550_ldc550_ws0_f32
//allotted time : 9999783.372704
//#trials to find viable hp in graph : 1 (0.076396 [ms]) 

//[1, 0.00s]	A_MIC8_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC8_PAD2_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR32_GAL1_PUN0_ICE1_IWI0_SZT0_MAD1_NAW64_UFO0_MAC64_SKW10_AFI1_MIA1
//compiling MAIN. Done in 0.671 [s]
//tt: 	 k0:	sum: 	 Gflops/s:
//0.344	 0.344	0.344	 966.434
//0.320	 0.320	0.320	 1041.445 (MEDIAN)
//0.320	 0.320	0.320	 1039.363
//0.303	 0.303	0.303	 1097.504
//0.304	 0.304	0.304	 1095.769

//[2, 0.68s]	


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




