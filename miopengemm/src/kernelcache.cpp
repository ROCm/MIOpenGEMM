/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <map>
#include <sstream>
#include <string>
#include <algorithm>
#include <miopengemm/enums.hpp>
#include <miopengemm/kernelcache.hpp>

namespace MIOpenGEMM
{

size_t CacheKeyHash::operator()(const CacheKey & ck) const{
  return __hash(ck.concatenated);
}
  
CacheKeyPresence KernelCache::check_for(const CacheKey& ckey) const
{
  if (vals.count(ckey) == 0){
    std::string open = "No cache entry from keys: " + ckey.get_string();
    std::string close = " (see gencache.cpp for example generating cache entry)";
    // TODO : how about the individual keys? Add this print functionality.
    return open + close;  
  }
  return {};
}


CacheKey::CacheKey(const std::string& dvc_,
                   const Constraints& cns_,
                   const Geometry& geo_)
  : dvc(dvc_), constraints(cns_), gg(geo_), concatenated(dvc_ + constraints.get_r_str() + gg.get_string())
{
}

std::string CacheKey::get_string() const
{
  std::stringstream ss;
  ss << "device       :   `" << dvc << "'\n";
  ss << "constraints  :   `" << constraints.get_r_str() << "'\n"; 
  ss << "geometry     :   `" << gg.get_string() << "'\n";
  return ss.str();
}

const KernelCache kernel_cache = get_kernel_cache();

KernelCache get_kernel_cache()
{
  KernelCache kc;
  #include "deepbench.cachetxt"
  return kc;
}

CachedSolution KernelCache::at(const CacheKey& ckey) const
{
  CacheKeyPresence ckp = check_for(ckey);
  if (!ckp.is_present)
  {
    throw miog_error(ckp.msg);
  }
  return vals.at(ckey);//.dvc).at(ckey.cns).at(ckey.geo);
}

void KernelCache::add(const CacheKey& ckey, const CachedSolution& tgcs)
{
  CacheKeyPresence ckp = check_for(ckey);
  if (ckp.is_present)
  {
    std::stringstream ss;
    ss << "Cannot add cache entry if one already exists, with. Keys: " << ckey.get_string()
       << "The existing entry is: " << at(ckey).get_string() << "The proposed entry is, "
       << tgcs.get_string() << "Please choose between these, manually remove existing if nec. ";
    throw miog_error(ss.str());
  }
  
  vals[ckey] = tgcs;
}

//CachedSolution get_generic_solution(const Constraints& constraints, const Geometry&  gg)
//{
  //throw miog_error("get_generic_cached_solution not impled");
//}

std::string CachedSolution::get_string() const
{
  std::unordered_map<CacheKey, std::string, CacheKeyHash> blob;
  std::stringstream ss;
  ss << "(hp) " << hp.get_string() << "\n";
  ss << "(stats) " << stats.get_string();
  return ss.str();
}


std::vector<CacheKey> KernelCache::get_keys() const{
  std::vector<CacheKey> keys;
  for (auto & x : vals){
    auto ck = std::get<0>(x);
    keys.push_back(ck);
  }
  return keys;
}
    
  
  //filtered(const std::vector<std::string> & device_frags, const std::vector<Geometry> & geometries) const{
  
  //std::vector<CacheKey> filtered;
  //for (auto & x : vals){
    //auto ck = std::get<0>(x);

    ////for  (auto & C : geometries){
      ////std::cout << C.get_string() << std::endl;
    ////}
    ////std::abort();

    
      //for (const auto & frag : device_frags){
        //if (ck.dvc.find(frag) != std::string::npos) {
          //if (std::find(filtered.begin(), filtered.end(), ck) == filtered.end()){
            //filtered.push_back(ck);
          //}
       //}
     //}
   //}
 //}
 //return filtered;
  
//}

void filter_device(std::vector<CacheKey> & cks, const std::vector<std::string> & device_frags){
  std::vector<CacheKey> valid;
  for (auto & ck : cks){
    for (const auto & frag : device_frags){
      if (ck.dvc.find(frag) != std::string::npos){
        valid.push_back(ck);
        break;
      }
    }
  }
  cks = std::move(valid);
}

void filter_geometries(std::vector<CacheKey> & cks, const std::vector<Geometry> & geometries){
  std::vector<CacheKey> valid;
  for (auto & ck : cks){
    if (std::find(geometries.begin(), geometries.end(), ck.gg) != geometries.end()){
      valid.push_back(ck);
    }
  }
  cks = std::move(valid);
}

void filter_floattype(std::vector<CacheKey> & cks, size_t float_size_bytes){
  std::vector<CacheKey> valid;
  for (auto & ck : cks){
    if (ck.gg.derived.float_size_bytes == float_size_bytes){
      valid.push_back(ck);
    }
  }
  cks = std::move(valid);
}




}
