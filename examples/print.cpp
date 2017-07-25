/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

// TODO : nform is still using size_t. change this back.

int main()
{
  using namespace MIOpenGEMM;
  //Geometry gg("tC0_tA0_tB1_colMaj1_m512_n512_k512_lda512_ldb512_ldc512_ws10000000_f32");
  Geometry gg("tC0_tA0_tB1_colMaj1_m4_n4_k2_lda32_ldb32_ldc32_ws10000000_f32");
  HyPas    hypas(
              {"MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW2",
               "MIC4_PAD0_PLU0_LIW0_MIW0_WOS0_VEW2",
               "UNR2_GAL1_PUN0_ICE1_NAW64_UFO0_MAC1_SKW10_IWI0_SZT0"});

               
  owrite::Writer  mowri(Ver::E::TERMINAL, "");
  kerngen::Bundle bundle(hypas, gg, mowri);

  for (auto& x : bundle.v_tgks)
  {
    // set this to the directory to write kernels to
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
