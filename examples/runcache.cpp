/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/geometries.hpp>

int main(){  
  using namespace MIOpenGEMM;

  Offsets offsets = get_padding_offsets();

  owrite::Writer mowri(Ver::E::TRACK, "");
  CLHint devhint;
  auto cache_keys = kernel_cache.get_filtered({""}, get_deepbench(0));
  
  for (auto & ck : cache_keys){
    auto soln = kernel_cache.at(ck);
    dev::Boa boa(ck.gg, offsets, mowri, devhint);  
    boa.benchgemm({soln.hp},{5, 10.});    
  }
  return 0;
}



