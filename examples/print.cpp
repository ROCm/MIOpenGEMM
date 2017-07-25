/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>

// TODO : nform is still using size_t. change this back.

int main()
{
  using namespace MIOpenGEMM;
 
  Geometry gg("tC0_tA0_tB1_colMaj1_m64_n192_k30_lda5000_ldb5000_ldc1000_ws10000000_f32");
  HyPas    hypas(
  "A_MIC8_PAD1_PLU0_LIW1_MIW1_WOS0_VEW4__B_MIC6_PAD2_PLU0_LIW0_MIW1_WOS0_VEW4__C_UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW16_UFO0_MAC256_SKW11");
               
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
