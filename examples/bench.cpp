/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/redirection.hpp>


MIOpenGEMM::HyPas get_redirected(const MIOpenGEMM::HyPas & hypas, bool swap_ab){

  using namespace MIOpenGEMM;

  if (swap_ab == false){
    return hypas;
  }

  else{
    auto c_sus = hypas.sus[Mat::E::C];
    c_sus.vs[NonChi::E::SKW] = 20 - c_sus.vs[NonChi::E::SKW]; 
    
    if (c_sus.vs[NonChi::E::GAL] == GroupAllocation::E::BYROW){
      c_sus.vs[NonChi::E::GAL] = GroupAllocation::E::BYCOL;
    }

    else if (c_sus.vs[NonChi::E::GAL] == GroupAllocation::E::BYCOL){
      c_sus.vs[NonChi::E::GAL] = GroupAllocation::E::BYROW;
    }    
    



    if (c_sus.vs[NonChi::E::AFI] == Binary::E::YES){
      c_sus.vs[NonChi::E::AFI] = Binary::E::NO;
    }

    else if (c_sus.vs[NonChi::E::AFI] == Binary::E::NO){
      c_sus.vs[NonChi::E::AFI] = Binary::E::YES;
    }



    if (c_sus.vs[NonChi::E::MIA] == MicroAllocation::E::BYA){
      c_sus.vs[NonChi::E::MIA] = MicroAllocation::E::BYB;
    }

    else if (c_sus.vs[NonChi::E::MIA] == MicroAllocation::E::BYB){
      c_sus.vs[NonChi::E::MIA] = MicroAllocation::E::BYA;
    }
    
    
        
    HyPas red {{hypas.sus[Mat::E::B], hypas.sus[Mat::E::A], c_sus }};
    return red;
  }
}

int main()
{
  using namespace MIOpenGEMM;

  // A freeze on compile case




Geometry gg("tC0_tA1_tB0_colMaj1_m8192_n256_k100_lda10000_ldb10000_ldc10000_ws0_f32");
HyPas hypas({ // hp
"MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1",
"MIC8_PAD1_PLU1_LIW1_MIW1_WOS0_VEW1",
"UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC64_SKW10_AFI0_MIA0"});

  bool swap_ab;
  auto gg_canonical = redirection::get_canonical(gg, swap_ab);
  gg_canonical.tX[Mat::E::C] = 0;
  HyPas hp_redirected = get_redirected(hypas, swap_ab);

  //auto gg_canonical = gg;
  //gg_canonical.tX[Mat::E::C] = 1;
  //HyPas hp_redirected = hypas;

  
  std::cout << "\n" << gg.get_string();
  std::cout << "\n" << gg_canonical.get_string() << "   (" << swap_ab << ")" << std::endl;
 
  std::cout << "\n" << hypas.get_string();
  std::cout << "\n" << hp_redirected.get_string() << "   (" << swap_ab << ")" << std::endl;
 
  CLHint         devhint;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa       boa(gg, offsets, mowri, devhint);
  dev::Boa       boa2(gg_canonical, offsets, mowri, devhint);




  for  (unsigned i = 0 ; i < 1; ++i){
    std::cout << "\n\n\n";
    boa.benchgemm({hypas}, {{0, 30}, {0., 100.}});
    boa2.benchgemm({hp_redirected}, {{0, 30}, {0., 100.}});
    
  }

  return 0;
}




