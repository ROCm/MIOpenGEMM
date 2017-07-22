/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <functional>
#include <unordered_map>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/kernelcachemerge.hpp>

namespace MIOpenGEMM{


std::vector<bool> get_thue_morse(size_t length){
  std::vector<bool> thue_morse {true, false};
  while (thue_morse.size() < length){
    for (auto x : thue_morse){
      thue_morse.push_back(!x);
    }
  } 
  thue_morse.resize(length);
  return thue_morse;
}

template <typename TFl>
void populate(const std::vector<CacheKey> & cache_keys, const KernelCache & kc1, const KernelCache & kc2, KernelCache & kc){
    
  Offsets        offsets = get_zero_offsets();
  owrite::Writer mowri(Ver::E::SILENT, "");
  CLHint         devhint;

  std::vector<Geometry> final_geometries;
  for (auto& x : cache_keys)
  {
    final_geometries.push_back(x.gg);
  }
  
  std::cout << "generating random matrices on CPU ... " << std::flush;
  // we set the CPU memory once for all geometries.
  // This is much faster than once for each geometry using Boas
  std::array<std::vector<TFl>, Mat::E::N> a_mem;
  std::vector<std::vector<TFl>*> v_mem{&a_mem[Mat::E::A], &a_mem[Mat::E::B], &a_mem[Mat::E::C]};
  setabcw::set_multigeom_abc<TFl>(v_mem, final_geometries, offsets);
  std::array<const TFl*, Mat::E::N> r_mem{
    a_mem[Mat::E::A].data(), a_mem[Mat::E::B].data(), a_mem[Mat::E::C].data()};
  std::cout << "done.\n" << std::endl;
  
  std::cout << "Will perform Thue–Morse (aka ABBA BAAB) 1-on-1." << std::endl;
  for (size_t i = 0; i < cache_keys.size(); ++i)
  {
    
    std::cout << '\n';
    
    auto ck = cache_keys[i];
    dev::Diva<TFl> diva1(ck.gg, offsets, r_mem, mowri, devhint);
    dev::Diva<TFl> diva2(ck.gg, offsets, r_mem, mowri, devhint);

    std::cout << ck.gg.get_string() << std::endl;
    std::cout << "soln1 : " << kc1.at(ck).get_string() << std::endl;
    std::cout << "soln2 : " << kc2.at(ck).get_string() << std::endl;
    
    std::string prefix = std::to_string(i) + "/" + std::to_string(cache_keys.size());
    prefix.resize(8, ' ');
    
    std::vector<double> times_kc1;
    std::vector<double> times_kc2;
    
    auto act_kcx = [&ck, &prefix](const KernelCache & kcx, std::string frag, std::vector<double> & times, dev::Diva<TFl> & diva){
      std::cout << prefix  << frag <<  "   ";
      times.push_back(diva.benchgemm({kcx.at(ck)}, {{0, 3}, {0, 0.5}}).back().back());
      std::cout << stringutil::get_char_padded(ck.gg.get_gflops(times.back()),8) << std::endl;
    };

    size_t kc1_wins = 0;
    size_t kc2_wins = 0;
    
    if (kc1.at(ck) == kc2.at(ck)){
      kc.add(ck, kc1.at(ck));
      std::cout << "(same soln)\n";
      continue;
    }
    
    for (auto kc1_first : get_thue_morse(5)){
      if (kc1_first){
        act_kcx(kc1, "++kc1", times_kc1, diva1);
        act_kcx(kc2, "--kc2", times_kc2, diva1);
      }
      else{
        act_kcx(kc2, "--kc2", times_kc2, diva1);
        act_kcx(kc1, "++kc1", times_kc1, diva1);
      }
      
      kc1_wins += (times_kc1.back() < times_kc2.back());
      kc2_wins += (times_kc2.back() < times_kc1.back());
      
    }
    
    if (kc1_wins > kc2_wins){
      std::cout << "kc1 won, " << kc1_wins << ':' << kc2_wins << '.' << '\n';
      kc.add(ck, kc1.at(ck));
    }
    else{
      std::cout << "kc2 won, " << kc2_wins << ':' << kc1_wins << '.' << '\n';
      kc.add(ck, kc2.at(ck));
    }
  }
}
  
KernelCache get_merged(const KernelCache & kc1, const KernelCache & kc2){

  KernelCache kc;
 
  std::map<char, std::vector<CacheKey>> in_both;
    
  size_t from_kc1 {0};
  size_t from_kc2 {0};
  size_t undetermined {0};
  for (auto & k1 : kc1.get_keys()){
    if (!kc2.check_for(k1).is_present){
      kc.add(k1, kc1.at(k1));
      ++from_kc1;
    }
    else{
      if (in_both.count(k1.gg.floattype) == 0){
        in_both[k1.gg.floattype] = {};
      }
      in_both[k1.gg.floattype].push_back(k1);
      ++undetermined;
    }
  }
  
  for (auto & k2 : kc2.get_keys()){
    if (!kc1.check_for(k2).is_present){
      kc.add(k2, kc2.at(k2));
      ++from_kc2;
    }
  }
  
  std::cout << "from kc1 : " << from_kc1 << ", from kc2 : " << from_kc2 << ", to be determined : " << undetermined << std::endl;


  for (auto & x : in_both){
    switch (std::get<0>(x)){
      case 'f': populate<float>(in_both['f'], kc1, kc2, kc); break;
      case 'd': populate<double>(in_both['f'], kc1, kc2, kc); break;
      default : throw miog_error("unrecognised floattype in get_merged");
    }
  }
  
  return kc;
}

  

}
