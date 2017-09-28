/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/redirection.hpp>
#include <miopengemm/tinytwo.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA1_tB0_colMaj1_m4001_n4001_k4001_lda4001_ldb4001_ldc4001_ws100000000_f32");
  HyPas    hp{{{"MIC6_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1",
             "MIC8_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1",
             "UNR16_GAL3_PUN1_ICE1_IWI0_SZT0_NAW16_UFO0_MAC64_SKW10_AFI0_MIA0_MAD1"}}};
  owrite::Writer  mowri(Ver::E::TERMINAL, "");
  kerngen::Bundle bundle(hp, gg);

  for (auto& x : bundle.v_tgks)
  {
    auto dirname = "/home/james/ptest/" + gg.get_string() + "/" + hp.get_contig_string() + "/";

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
