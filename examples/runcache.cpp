/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/geometries.hpp>
#include <miopengemm/setabcw.hpp>


template <typename TFl>
int runcache_v2(bool only_deepbench, bool all_devices){
  
  using namespace MIOpenGEMM;
  Offsets offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::MULTIBENCH, "");
  CLHint devhint;
  
  auto cache_keys = kernel_cache.get_keys();
  if (!all_devices){
    owrite::Writer silent_mowri(Ver::E::TERMINAL, "");
    oclutil::DevInfo devinfo(devhint, silent_mowri);
    filter_device(cache_keys, {devinfo.device_name});
  }
  
  if (only_deepbench){
    filter_geometries(cache_keys, get_deepbench(0));
  }
  filter_floattype(cache_keys, sizeof(TFl));
  std::vector<Geometry> final_geometries;
  for (auto & x : cache_keys){
    final_geometries.push_back(x.gg);
  }
  
  // we set the CPU memory once for all geometries. 
  // This is much faster than once for each geometry using Boas
  std::array<std::vector<TFl>, Mat::E::N> a_mem;
  std::vector<std::vector<TFl> *> v_mem {&a_mem[Mat::E::A], &a_mem[Mat::E::B], &a_mem[Mat::E::C]};
  setabcw::set_multigeom_abc<TFl>(v_mem, final_geometries, offsets);
  std::array<const TFl *, Mat::E::N> r_mem {a_mem[Mat::E::A].data(), a_mem[Mat::E::B].data(), a_mem[Mat::E::C].data()};
  
  for (size_t i = 0; i < cache_keys.size(); ++ i){
    auto ck = cache_keys[i];
    auto soln = kernel_cache.at(ck);    
    if (ck.gg.floattype == 'f'){
      dev::Diva<TFl> diva(ck.gg, offsets, r_mem, mowri, devhint);
      std::string prefix = std::to_string(i) + "/" + std::to_string(cache_keys.size());
      prefix.resize(7, ' ');
      std::cout << prefix;
      diva.benchgemm({soln.hp},{5, 10.});    
    }
  }
  
  return 0;
}


int main(int argc, char * argv []){
  
  std::vector<std::string> sargs;
  for (size_t i = 1; i < argc; ++i){
    sargs.push_back(argv[i]);
  }
 
  bool only_deepbench = false;
  bool all_devices = false;
 
  
  for (auto & x : sargs){
    if (x == "D"){
      only_deepbench = true;
    }
    
    if (x == "A"){
      all_devices = true;
    }
    
    else{
      std::stringstream errm;
      errm << "unrecognised flag ";
      errm << x << '.';
      errm << " accepted flags are\n";
      errm << "'D' (DeepBench geometries : only benchmark DeepBench geometries) and\n";
      errm << "'A' (All devices : if a cache entry is for another device, run anyway). ";
      throw MIOpenGEMM::miog_error(errm.str());
    }
  }

  return runcache_v2<float>(only_deepbench, all_devices);
}

