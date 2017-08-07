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
int runcache_v2(std::map<char, std::vector<std::string>> & filters)
{

  
  using namespace MIOpenGEMM;
  
  Offsets offsets = get_zero_offsets();  
  auto cache_keys = kernel_cache.get_keys();

  CLHint devhint;
  if (std::count(filters['d'].begin(), filters['d'].end(), "a") == 0)
  {
    std::vector<std::string> dev_filters;
    for (auto & x : filters['d']){
      if (x == "d"){
        owrite::Writer silent_mowri(Ver::E::TERMINAL, "");
        oclutil::DevInfo devinfo(devhint, silent_mowri);
        dev_filters.push_back(devinfo.device_name);
      }
      else{
        dev_filters.push_back(x);
      }
    }
    std::cout << "Will filter cache for devices:   {";
    for (auto & x : dev_filters){
      std::cout << "  " << x << "  ";
    }
    std::cout << "}\n"; 
    filter_device(cache_keys, dev_filters);
  }
  else{
    std::cout << "Will consider all devices in cache\n";
  }

  if (std::count(filters['g'].begin(), filters['g'].end(), "a") == 0){

    
    std::vector<std::string> geometries_to_filter;
    bool filter_db = false;
    std::vector<Geometry> dbgs;

    if (std::count(filters['g'].begin(), filters['g'].end(), "d") != 0){
    
      {
        dbgs = get_deepbench(0);
        filter_db = true;
        geometries_to_filter.push_back("deepbench");
      }
    }
    
    bool filter_square = false;
    if (std::count(filters['g'].begin(), filters['g'].end(), "s") != 0){
      filter_square = true;
      geometries_to_filter.push_back("square");
    }
    
    std::cout << "Will filter cache for geometries:  {";
    for (auto & x : geometries_to_filter){
      std::cout << "  " << x << "  ";
    }
    std::cout << "}\n"; 

  

    auto all_geometries = get_geometries(cache_keys);
    std::vector<Geometry> filtered_geometries;
    for (auto & x : all_geometries){
      if (filter_square && x.m == x.n && x.m == x.k){
        filtered_geometries.push_back(x);
      }
      else if (filter_db && std::find(dbgs.begin(), dbgs.end(), x) != dbgs.end()){
        filtered_geometries.push_back(x);        
      }      
    }
    filter_geometries(cache_keys, filtered_geometries);
  }
  else{
    std::cout << "Will consider all geometries in cache\n";
  }
  
  if (cache_keys.size() == 0){
    std::cout << "No cache keys remain after filtering. \n";
    std::cout << "Note that all device keys in the cache are  {";
    for (auto & x : get_devices(kernel_cache.get_keys())){
      std::cout << "  " << x << "  ";
    }
    std::cout << "}\n";
    return -1;
  }
  

  filter_floattype(cache_keys, sizeof(TFl));
  std::cout << "generating random matrices on CPU ... " << std::flush;
  setabcw::CpuMemBundle<TFl> cmb(get_geometries(cache_keys), offsets);
  std::cout << "done.\n" << std::endl;

  if (std::count(filters['w'].begin(), filters['w'].end(), "a") != 0){
    std::cout << "\nPerforming accuracy tests\n\n";
    owrite::Writer mowri(Ver::E::TERMINAL, "");
    for (size_t i = 0; i < cache_keys.size(); ++i)
    {
      auto ck = cache_keys[i];
      if (ck.gg.floattype == 'f')
      {
        dev::Diva<TFl> diva(ck.gg, offsets, cmb.r_mem, mowri, devhint);
        diva.accuracy_test(kernel_cache.at(ck, redirection::get_is_not_canonical(ck.gg)), nullptr);     
        mowri << "\n\n";
      }
    }
  }
    
  
  if (std::count(filters['w'].begin(), filters['w'].end(), "b") != 0){
    std::cout << "\nBenchmarking\n\n";
    owrite::Writer mowri(Ver::E::MULTIBENCH, "");
    for (size_t i = 0; i < cache_keys.size(); ++i)
    {
      auto ck = cache_keys[i];
      if (ck.gg.floattype == 'f')
      {
        dev::Diva<TFl> diva(ck.gg, offsets, cmb.r_mem, mowri, devhint);
        std::string prefix = std::to_string(i) + "/" + std::to_string(cache_keys.size());
        prefix.resize(8, ' ');
        std::cout << prefix << " ";
        diva.benchgemm({kernel_cache.at(ck, redirection::get_is_not_canonical(ck.gg))}, {{0, 5}, {0, 0.15}});      
      }
    }
  }
  

  return 0;
}

int main(int argc, char* argv[])
{
  std::map<char, std::string> args = {
    {'g', "geometries to look for in cache"},
    {'d', "devices to look for in cache"},
    {'w', "what to do"}
  };
  
  std::map<char, std::vector<std::string>> options = {
    {'g', {"a (all)", "d (deepbench)", "s (m=n=k)"}},
    {'d', {"a (all)","d (default)", "any other string"}},
    {'w', {"b (benchmark)", "a (accuracy)"}}
  };
  
  
  std::map<char, std::string> defaults = {
    {'g', "d"},
    {'d', "d"},
    {'w', "b"}    
  };
  
  // TODO: confirm that above have same keys.
  
  std::stringstream hss;
  hss << "\n\n ";
  std::map<char, std::vector<std::string>> filters;
  //std::set<char> keys;
  for (auto & x : args){
    hss << '-' << x.first << " : " << x.second << '\n'; 
    hss << "options : ";
    for (auto & o : options[x.first]){
      hss << " " << o << " ";
    }
    hss << "\ndefault :  " << defaults[x.first] << "\n\n";
    filters[x.first] = {};
    //set.insert(x);
  }
  hss << "\nexamples:\n";
  hss << "`runcache -g d -d a -w b`\n";
  hss << " benchmarks all cache entries with a deepbench geometry \n";
  hss << "`runcache -g a -d gfx803`\n";
  hss << " benchmarks of all cache entries which match device gfx803 \n";
  hss << "`runcache -w b a`\n";
  hss << " benchmarks and accuracy test of all cache entries \n";

  std::string help = hss.str();

  using namespace MIOpenGEMM;

  if (argc % 2 != 1){
    throw miog_error("Odd number of keys+vals is incorrect.\n" + help);
  }

  std::vector<std::string> parsed;
  for (int i = 1; i < argc; ++ i){
    parsed.push_back(argv[i]);
  }
  
  char key = '?';
  for (auto & x : parsed){
    if (x.size() == 2 && x.compare(0,1,"-") == 0){
      key = x[1];
    }
    else{
      if (filters.count(key) == 0){
        std::stringstream errm;
        errm << "Unrecognised key\n" <<  help;
        throw miog_error(errm.str());
      }
      else{
        filters[key].push_back(x);
      }
    }
  }

  for (auto x : defaults){
    if (filters[x.first].size() == 0){
      filters[x.first] = {x.second};
    }
  }

  return runcache_v2<float>(filters);
}


