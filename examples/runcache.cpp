/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/geometries.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/kernelcachemerge.hpp>
#include <miopengemm/setabcw.hpp>

  
template <typename TFl>
int runcache_v2(bool only_deepbench, bool all_devices)
{

  using namespace MIOpenGEMM;
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::MULTIBENCH, "");
  CLHint         devhint;

  auto cache_keys = kernel_cache.get_keys();
  if (!all_devices)
  {
    owrite::Writer silent_mowri(Ver::E::TERMINAL, "");
    oclutil::DevInfo devinfo(devhint, silent_mowri);
    filter_device(cache_keys, {devinfo.device_name});
  }

  if (only_deepbench)
  {
    filter_geometries(cache_keys, get_deepbench(0));
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
      std::cout << ck.gg.get_string() << '\n'; 
      std::cout << kernel_cache.at(ck).get_string() << '\n';
      std::cout << prefix << " ";
      diva.benchgemm({kernel_cache.at(ck)}, {{0, 3}, {0, 10.}});
      std::cout << '\n';
      
    }
  }
  return 0;
}

int main(int argc, char* argv[])
{

  using namespace MIOpenGEMM;
  std::vector<std::string> sargs;
  for (size_t i = 1; i < argc; ++i)
  {
    sargs.push_back(argv[i]);
  }

  bool only_deepbench = false;
  bool all_devices    = false;

  for (auto& x : sargs)
  {
    if (x == "D")
    {
      only_deepbench = true;
    }

    if (x == "A")
    {
      all_devices = true;
    }

    else
    {
      std::stringstream errm;
      errm << "unrecognised flag ";
      errm << x << '.';
      errm << " accepted flags are\n";
      errm << "'D' (DeepBench geometries : only benchmark DeepBench geometries) and\n";
      errm << "'A' (All devices : if a cache entry is for another device, run anyway). ";
      throw miog_error(errm.str());
    }
  }
  
  

  //auto kcn = get_merged(kernel_cache, kernel_cache2);


  //std::ofstream floper("/home/james/kc33.txt", std::ios::out);
    
  ////kcn.write()  
  //for (auto & ck : kcn.get_keys()){
    //std::cout << ck.get_string() << std::endl;
    //floper << '\n' << get_cache_entry_string(ck, kcn.at(ck));
  //}

  //floper.close();
  
  
  //return 0;
  return runcache_v2<float>(only_deepbench, all_devices);
}


