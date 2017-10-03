/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/redirection.hpp>
#include <miopengemm/tinytwo.hpp>



int main()
{
  using namespace MIOpenGEMM;
  
  
  Geometry gg{"tC0_tA0_tB1_colMaj1_m1200_n1200_k1200_lda1200_ldb1200_ldc1200_ws0_f32"};

  HyPas hp{{{ //hp
"MIC8_PAD1_PLU0_LIW0_MIW1_WOS0_LOM0_VEW2",
"MIC6_PAD1_PLU0_LIW0_MIW0_WOS0_LOM1_VEW2",
"UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC64_SKW10_AFI0_MIA0"}}};


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
