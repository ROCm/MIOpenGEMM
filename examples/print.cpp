/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

// TODO : nform is still using size_t. change this back.

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA0_tB0_colMaj0_m28_n2048_k2048_lda2048_ldb2048_ldc2048_ws5000000_f32");
  HyPas    hypas({"MIC1_PAD1_PLU0_LIW0_MIW1_WOS2",
               "MIC8_PAD1_PLU0_LIW0_MIW1_WOS2",
               "UNR8_GAL3_PUN1_ICE1_NAW64_UFO1_MAC64_SKW10"});

  owrite::Writer  mowri(Ver::E::TERMINAL, "");
  kerngen::Bundle bundle(hypas, gg, mowri);

  for (auto& x : bundle.v_tgks)
  {
    // set this to the directory to write kernels to
    auto dirname = "/home/james/test/" + gg.get_string() + "/" + hypas.get_string() + "/";
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
