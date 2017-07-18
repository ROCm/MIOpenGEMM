/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <map>
#include <sstream>
#include <string>
#include <miopengemm/enums.hpp>
#include <miopengemm/kernelcache.hpp>

namespace MIOpenGEMM
{


size_t CacheKeyHash::operator()(const CacheKey & ck) const{
  return __hash(ck.get_concatenated());
}
  
CacheKeyPresence KernelCache::check_for(const CacheKey& ckey) const
{
  std::string open = "No cache entry from keys: " + ckey.get_string();
  std::string close = " (see gencache.cpp for example generating cache entry)";
  // TODO : how about the individual keys?
  
  if (vals.count(ckey) == 0){
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
//#include "cacheexample.cachetxt"

kc.add(
// cache key (device, constraint, geometry)
{"Fiji", 
{"A_MIC8"}, 
{"tC0_tA0_tB0_colMaj1_m5000_n5000_k5000_lda5000_ldb5000_ldc5000_ws1_f32"}},
// cache solution (hyperparam, statistics)
{{{
"MIC8_PAD2_PLU0_LIW0_MIW1_WOS0",
"MIC6_PAD1_PLU0_LIW1_MIW0_WOS0",
"UNR8_GAL2_PUN0_ICE1_NAW64_UFO0_MAC256_SKW10"}},           
{59.2006, 4222.93, 3.32959, "Sun May 14 12:29:44 2017", {3, 2, 1, 1e12, SummStat::E::MAX}}
});

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
       << tgcs.get_string()
       << "Please choose between these, and manually remove existing one if nec. ";
    throw miog_error(ss.str());
  }
  
  vals[ckey] = tgcs;
}

CachedSolution get_generic_solution(const Constraints& constraints, const Geometry&  gg)
{
  throw miog_error("get_generic_cached_solution not impled");
}

std::string CachedSolution::get_string() const
{
  
  std::unordered_map<CacheKey, std::string, CacheKeyHash> blob;
  
  std::stringstream ss;
  ss << "(hp) " << hp.get_string() << "\n";
  ss << "(stats) " << stats.get_string();
  return ss.str();
}
}
