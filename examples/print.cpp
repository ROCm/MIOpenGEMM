/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/redirection.hpp>
#include <miopengemm/tinytwo.hpp>



int main()
{
  using namespace MIOpenGEMM;



  Geometry gg{"tC0_tA0_tB1_colMaj1_m6004_n6004_k6004_lda6004_ldb6004_ldc6004_ws500_ws50000001_ws50000000_f32"};

  HyPas hp{{{ //hp
  
  "MIC8_PAD1_PLU0_LIW0_MIW0_WOS1_LOM1_VEW1",
  "MIC6_PAD1_PLU0_LIW0_MIW1_WOS1_LOM1_VEW1",
  "UNR16_GAL3_PUN0_ICE1_IWI1_SZT0_MAD0_NAW16_UFO0_MAC256_SKW10_AFI0_MIA0_PAK0_STR0"}}};

   owrite::Writer  mowri(Ver::E::TERMINAL, "");
  kerngen::Bundle bundle(hp, gg);

    auto dirname = "/home/james/ptest/" + gg.get_string() + "/" + hp.get_contig_string() + "/";
    // WARNING : mkdir only works on linux/mac
    std::string syscall = "mkdir -p " + dirname;
    std::system(syscall.c_str());


  for (auto& x : bundle.v_tgks)
  {
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
