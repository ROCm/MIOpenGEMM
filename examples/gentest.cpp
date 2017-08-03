/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/standalone.hpp>

int main()
{
  using namespace MIOpenGEMM;
  Geometry gg("tC0_tA1_tB0_colMaj0_m300_n300_k300_lda300_ldb300_ldc300_ws0_f32");  
  HyPas    hp("A_MIC1_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__C_UNR16_GAL2_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");  
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  auto standalone_source = standalone::make(gg, hp, mowri);
  
  auto fname = "/home/james//miopengemm/MIOpenGEMM/tests/rocmaccu.cpp";
  mowri << "writing " << fname << " ... " << Flush;
  std::ofstream floper(fname, std::ios::out);
  floper << standalone_source;
  floper.close();
  mowri << "done." << Endl;


  return 0;
}
