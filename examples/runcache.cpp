/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/geometries.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/kernelcachemerge.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/setabcw.hpp>
  
template <typename TFl>
int runcache_v2(char geom_filter, bool all_devices)
{

  using namespace MIOpenGEMM;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::MULTIBENCH, "");
  auto cache_keys = kernel_cache.get_keys();

    CLHint devhint;

  
  if (!all_devices)
  {
    owrite::Writer silent_mowri(Ver::E::TERMINAL, "");
    oclutil::DevInfo devinfo(devhint, silent_mowri);
    filter_device(cache_keys, {devinfo.device_name});
  }

  if (geom_filter == 'D')
  {
    filter_geometries(cache_keys, get_deepbench(0));
  }
  
  else if (geom_filter == 'S'){
    auto all_geometries = get_geometries(cache_keys);
    std::vector<Geometry> square_geometries;
    for (auto & x : all_geometries){
      if (x.m == x.n && x.m == x.k){
        square_geometries.push_back(x);
      }
    }
    filter_geometries(cache_keys, square_geometries);
  }
  
  
  filter_floattype(cache_keys, sizeof(TFl));
  std::cout << "generating random matrices on CPU ... " << std::flush;
  setabcw::CpuMemBundle<TFl> cmb(get_geometries(cache_keys), offsets);
  std::cout << "done.\n" << std::endl;

  for (size_t i = 0; i < cache_keys.size(); ++i)
  {
    auto ck = cache_keys[i];
    if (ck.gg.floattype == 'f')
    {
      dev::Diva<TFl> diva(ck.gg, offsets, cmb.r_mem, mowri, devhint);
      std::string prefix = std::to_string(i) + "/" + std::to_string(cache_keys.size());
      prefix.resize(8, ' ');
      std::cout << prefix << " ";
      diva.benchgemm({kernel_cache.at(ck, redirection::get_is_not_canonical(ck.gg))}, {{0, 5}, {0, 0.1}});      
    }
  }
  return 0;
}

int main(int argc, char* argv[])
{

  using namespace MIOpenGEMM;

  if (argc % 2 != 1){
    throw miog_error("Odd number of args, not correct");
  }


  std::vector<std::vector<std::string>> keysvals (2);
  

  for (size_t i = 1; i < argc; ++i)
  {
    keysvals[(i-1)%2].push_back(argv[i]);
  }


  char geom_filter = 'Z';
  bool all_devices = false;

  
  for (size_t i = 0; i < keysvals[0].size(); ++i){
    auto key = keysvals[0][i];
    auto val = keysvals[1][i];

    if (key == "--device" || key == "-d"){
      if (val == "a"){
        all_devices = true;
      }
      else{
        throw miog_error("unregnised value for flag " + key + " Should be one of [a (for all)]");         
      }
    }
    
    else if (key == "--geometry" || key == "-g"){
      geom_filter = val[0];
      if (geom_filter != 'D' && geom_filter != 'S'){
        throw miog_error("unregnised value for flag " + key + " Should be one of [D (deepbench) and S (square)]");                 
      }
    }
    
    else{
      throw miog_error("allowed flags are --device (-d) and --geometry (-g)"); 
    }
  
  //for (auto& x : sargs)
  //{
    //if (x == "D" || x == "-D")
    //{
      //only_deepbench = true;
    //}

    //else if (x == "A" || x == "-A")
    //{
      //all_devices = true;
    //}

    //else
    //{
      //std::stringstream errm;
      //errm << "unrecognised flag ";
      //errm << x << '.';
      //errm << " accepted flags are\n";
      //errm << "'D' (DeepBench geometries : only benchmark DeepBench geometries) and\n";
      //errm << "'A' (All devices : if a cache entry is for another device, run anyway). ";
      //throw miog_error(errm.str());
    //}
  }
  
  return runcache_v2<float>(geom_filter, all_devices);
}


