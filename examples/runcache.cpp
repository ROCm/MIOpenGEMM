/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/kernelcache.hpp>
  

int main(){  
  using namespace MIOpenGEMM;


  auto bla = kernel_cache.check_for({"gfx803",
"A_WOS0__B_WOS0",
"tC0_tA0_tB0_colMaj0_m28_n2048_k2048_lda2048_ldb2048_ldc2048_ws5000000_f32",
""});

  std::cout << bla.msg;
  return 0;
}


  //std::string dvc;  // device
  //std::string cns;  // constraint
  //std::string geo;  // geometry
  //std::string cmm;  // comment
  //CacheKey(const std::string&, const std::string&, const std::string&, const std::string&);


  //Geometry gg();
  //Constraints constraints("A_MIC1_PAD1_PLU0_LIW0_MIW1_WOS2__C_ICE3");
  //FindParams find_params(100, 2., 3, 1., SummStat::E::MEDIAN);
  //Offsets offsets = get_padding_offsets();
  //owrite::Writer mowri(Ver::E::TRACK, "");
  //CLHint devhint;
  //dev::Boa boa(gg, offsets, mowri, devhint);  
  //Solution soln = boa.find(find_params, constraints);    
  
  //std::cout << " \n\n\nThe following string can be cut and paste into kernelcache.cpp";
  //std::cout << " \n\n-- snip -- -- -- snip --\n\n";
  //std::cout << soln.get_cache_entry_string();
  //std::cout << " -- snip -- -- -- snip --\n\n";
