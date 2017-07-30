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
 
  Geometry gg("tC0_tA0_tB0_colMaj1_m512_n512_k128_lda512_ldb5000_ldc512_ws0_f64");
  HyPas hypas("A_MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1__C_UNR8_GAL2_PUN1_ICE1_IWI1_SZT0_NAW16_UFO1_MAC64_SKW10_AFI1_MIA0");
  owrite::Writer  mowri(Ver::E::TERMINAL, "");
  kerngen::Bundle bundle(hypas, gg, mowri);

  for (auto& x : bundle.v_tgks)
  {
    auto dirname = "/home/james/ptest/" + gg.get_string() + "/" + hypas.get_string() + "/";
    
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
