/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/redirection.hpp>

// TODO : nform is still using size_t. change this back.


int main()
{
  using namespace MIOpenGEMM;



  //Geometry gg("tC1_tA1_tB1_colMaj1_m45_n56_k64_lda64_ldb65_ldc67_ws1_f32");    
  Geometry gg("tC1_tA1_tB1_colMaj1_m45_n56_k64_lda64_ldb64_ldc64_ws1_f32");    
  // AFI1 : no break. AFI0 : break.  
  HyPas    hp("A_MIC2_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL3_PUN1_ICE2_IWI0_SZT0_NAW16_UFO0_MAC1_SKW10_AFI0_MIA0");  


  owrite::Writer  mowri(Ver::E::TERMINAL, "");
  kerngen::Bundle bundle(hp, gg, mowri);

  for (auto& x : bundle.v_tgks)
  {
    auto dirname = "/home/james/ptest/" + gg.get_string() + "/" + hp.get_string() + "/";
    
    // WARNING : mkdir only works on linux/mac
    std::string syscall = "mkdir -p " + dirname;
    std::system(syscall.c_str());
    auto fname = dirname + x.kuses.full + ".cl";
    mowri << "writing " << fname << " ... " << Flush;
    std::ofstream floper(fname, std::ios::out);
    floper << x.kernstr;
    floper.close();
    mowri << "done." << Endl;
  }

  mowri << "\ndone all.\n";
  return 0;
}
