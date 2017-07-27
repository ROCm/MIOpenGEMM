/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <miopengemm/enums.hpp>
#include <miopengemm/kernelcache.hpp>

namespace MIOpenGEMM
{

size_t CacheKeyHash::operator()(const CacheKey& ck) const { return __hash(ck.concatenated); }


std::vector<Geometry> get_geometries(const std::vector<CacheKey> & cks){
  std::vector<Geometry> geometries;
  for (const auto& x : cks)
  {
    geometries.push_back(x.gg);
  }
  return geometries;  
}

CacheKeyPresence KernelCache::check_for(const CacheKey& ckey) const
{
  if (vals.count(ckey) == 0)
  {
    std::string open  = "No cache entry from keys: " + ckey.get_string();
    std::string close = " (see gencache.cpp for example generating cache entry)";
    // TODO : how about the individual keys? Add this print functionality.
    return open + close;
  }
  return {};
}

CacheKey::CacheKey(const std::string& dvc_, const Constraints& cns_, const Geometry& geo_)
  : dvc(dvc_),
    constraints(cns_),
    gg(geo_),
    concatenated(dvc_ + constraints.get_string() + gg.get_string())
{
}


std::string get_cache_entry_string(const CacheKey & ck, const HyPas & hypas)
{
  std::stringstream cache_write_ss;
  cache_write_ss << "kc.add(\n";
  cache_write_ss << "{\"" << ck.dvc << "\",  // dev\n";
  cache_write_ss << "{\"" << ck.constraints.get_string() << "\"},  // con\n";
  cache_write_ss << "{\"" << ck.gg.get_string() << "\"}}, // gg\n";
  cache_write_ss << "{{ // hp\n";
  cache_write_ss << "\"" << hypas.sus[Mat::E::A].get_string() << "\",\n";
  cache_write_ss << "\"" << hypas.sus[Mat::E::B].get_string() << "\",\n";
  cache_write_ss << "\"" << hypas.sus[Mat::E::C].get_string() << "\"}});\n";
  return cache_write_ss.str();
}



std::string CacheKey::get_string() const
{
  std::stringstream ss;
  ss << "device       :   `" << dvc << "'\n";
  ss << "constraints  :   `" << constraints.get_string() << "'\n";
  ss << "geometry     :   `" << gg.get_string() << "'\n";
  return ss.str();
}

KernelCache get_kernel_cache()
{
  KernelCache kc;
  #include "deepbench.cachetxt"
  //#include "cacheentries_temp.cachetxt"
  return kc;
}
const KernelCache kernel_cache = get_kernel_cache();




HyPas KernelCache::at(const CacheKey& ckey) const
{
  CacheKeyPresence ckp = check_for(ckey);
  if (!ckp.is_present)
  {
    throw miog_error(ckp.msg);
  }
  return vals.at(ckey);
}

void KernelCache::add(const CacheKey& ckey, const HyPas& hp)
{
  CacheKeyPresence ckp = check_for(ckey);
  if (ckp.is_present)
  {
    std::stringstream ss;
    ss << "Cannot add cache entry if one already exists, with. Keys: " << ckey.get_string()
       << "The existing entry is: " << at(ckey).get_string()
       << ". Please choose between these, manually remove existing if nec. ";
    throw miog_error(ss.str());
  }

  vals[ckey] = hp;
}

std::vector<CacheKey> KernelCache::get_keys() const
{
  std::vector<CacheKey> keys;
  for (auto& x : vals)
  {
    auto ck = std::get<0>(x);
    keys.push_back(ck);
  }
  return keys;
}

void filter_device(std::vector<CacheKey>& cks, const std::vector<std::string>& device_frags)
{
  std::vector<CacheKey> valid;
  for (auto& ck : cks)
  {
    for (const auto& frag : device_frags)
    {
      if (ck.dvc.find(frag) != std::string::npos)
      {
        valid.push_back(ck);
        break;
      }
    }
  }
  cks = std::move(valid);
}

void filter_geometries(std::vector<CacheKey>& cks, const std::vector<Geometry>& geometries)
{
  std::vector<CacheKey> valid;
  for (auto& ck : cks)
  {
    if (std::find(geometries.begin(), geometries.end(), ck.gg) != geometries.end())
    {
      valid.push_back(ck);
    }
  }
  cks = std::move(valid);
}

void filter_floattype(std::vector<CacheKey>& cks, size_t float_size_bytes)
{
  std::vector<CacheKey> valid;
  for (auto& ck : cks)
  {
    if (ck.gg.derived.float_size_bytes == float_size_bytes)
    {
      valid.push_back(ck);
    }
  }
  cks = std::move(valid);
}

double CacheKey::get_distance(const CacheKey& ck) const
{
  double distance = 0;
  distance += gg.get_distance(ck.gg);
  distance += 1e-6 * (dvc != ck.dvc);
  // constraints ? will be non-sym. TODO

  return distance;
}

bool KernelCache::nearest_derivable_is_within(const CacheKey& ck, double threshold) const
{
  for (auto& key : get_keys())
  {
    if (key.get_distance(ck) < threshold && Derivabilty(vals.at(key), ck.gg).is_derivable)
    {
      return true;
    }
  }
  return false;
}

CacheKey KernelCache::get_nearest_derivable(const CacheKey& ck) const
{
  if (!nearest_derivable_is_within(ck, std::numeric_limits<double>::max()))
  {
    throw miog_error("In get_nearest_derivable, with none within radius <double>::max.");
  }
  double d_nearest_derivable = std::numeric_limits<double>::max();

  auto cache_keys = get_keys();
  if (cache_keys.size() == 0)
  {
    throw miog_error("No cache keys. This is very strange.");
  }

  CacheKey nearest_derivable(cache_keys.back());

  for (auto& key : cache_keys)
  {
    if (ck.get_distance(key) < d_nearest_derivable && Derivabilty(vals.at(key), ck.gg).is_derivable)
    {
      d_nearest_derivable = ck.get_distance(key);
      nearest_derivable   = key;
    }
  }

  // confirm derivability
  if (!Derivabilty(vals.at(nearest_derivable), ck.gg).is_derivable)
  {
    throw miog_error("internal logic error : the nearest derivable is not derivable.");
  }
  return nearest_derivable;
}




  
}
