/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

// TODO : nform is still using size_t. change this back.

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA0_tB0_colMaj0_m32_n32_k16_lda512_ldb2048_ldc2048_ws5000000_f32");
  HyPas    hypas(
              {"MIC2_PAD1_PLU0_LIW0_MIW1_WOS0",
               "MIC2_PAD1_PLU0_LIW0_MIW1_WOS0",
               "UNR8_GAL1_PUN0_ICE2_NAW64_UFO0_MAC64_SKW10_IWI1"});
               
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
