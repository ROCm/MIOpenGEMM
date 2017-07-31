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
 
  Geometry gg("tC0_tA0_tB1_colMaj1_m125_n25_k255_lda131_ldb25_ldc1250_ws0_f32");  
  HyPas    hp("A_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW2__B_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL3_PUN1_ICE1_IWI0_SZT0_NAW16_UFO1_MAC4_SKW10_AFI1_MIA0");  
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
