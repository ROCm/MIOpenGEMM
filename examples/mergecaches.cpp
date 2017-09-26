/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <miopengemm/findparams.hpp>
#include <miopengemm/kernelcachemerge.hpp>

MIOpenGEMM::KernelCache get_kernel_cache2()
{
  MIOpenGEMM::KernelCache kc;
  //#include "/home/james/test48/cacheentries.txt"
  return kc;
}

int main()
{
  using namespace MIOpenGEMM;

  KernelCache kernel_cache2 = get_wSpaceReduced(get_kernel_cache2());

  for (auto ck : kernel_cache2.get_keys())
  {
    std::cout << ck.gg.get_string() << std::endl;
  }

  owrite::Writer mowri(Ver::E::MERGE, "");

  auto&& kernel_cache = get_kernel_cache();

  Halt          halt = {{{0, 5}}, {{0, 0.1}}};
  auto          kcn  = get_merged(kernel_cache, kernel_cache2, halt, mowri);
  std::ofstream floper("/home/james/test48/merged_cache48.txt", std::ios::out);
  for (auto& ck : kcn.get_keys())
  {
    floper << '\n' << kcn.get_cache_entry_string(ck);
  }
  floper.close();
  return 0;
}
