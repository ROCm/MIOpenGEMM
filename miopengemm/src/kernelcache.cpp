/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <miopengemm/enums.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/redirection.hpp>

namespace MIOpenGEMM
{

size_t CacheKeyHash::operator()(const CacheKey& ck) const { return __hash(ck.concatenated); }

std::vector<Geometry> get_geometries(const std::vector<CacheKey>& cks)
{
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
    return open + close;
  }
  return {};
}

CacheKey::CacheKey(const std::string& dvc_, const Constraints& cns_, const Geometry& geo_)
  : from_non_canonical(redirection::get_is_not_canonical(geo_)),
    dvc(dvc_),
    constraints(cns_.get_reflected(from_non_canonical)),
    gg(redirection::get_canonical(geo_)),
    concatenated(dvc_ + constraints.get_string() + gg.get_string())
{
}

std::string get_cache_entry_string(const CacheKey& ck, const HyPas& hypas, bool swap_ab)
{
  std::string       swap_ab_str = swap_ab ? "true" : "false";
  std::stringstream cache_write_ss;
  cache_write_ss << "kc.add(\n";
  cache_write_ss << "{\"" << ck.dvc << "\",  // dev\n";
  cache_write_ss << "{\"" << ck.constraints.get_string() << "\"},  // con\n";
  cache_write_ss << "{\"" << ck.gg.get_string() << "\"}}, // gg\n";
  cache_write_ss << "{{{ // hp\n";
  auto hp = hypas.get_reflected(swap_ab);
  cache_write_ss << "\"" << hp.sus[Mat::E::A].get_string() << "\",\n";
  cache_write_ss << "\"" << hp.sus[Mat::E::B].get_string() << "\",\n";
  cache_write_ss << "\"" << hp.sus[Mat::E::C].get_string() << "\"}}});\n";
  return cache_write_ss.str();
}

std::string KernelCache::get_cache_entry_string(const CacheKey& ck) const
{
  return MIOpenGEMM::get_cache_entry_string(ck, at(ck, false), false);
}

std::string CacheKey::get_string() const
{
  std::stringstream ss;
  ss << "device       :   `" << dvc << "'\n";
  ss << "constraints  :   `" << constraints.get_string() << "'\n";
  ss << "geometry     :   `" << gg.get_string() << "'\n";
  return ss.str();
}

KernelCache init_kernel_cache()
{
  KernelCache kc;

#include "cache1.cachetxt"
#include "cache2.cachetxt"
#include "cache3.cachetxt"
#include "cache4.cachetxt"
  return kc;
}

const KernelCache& get_kernel_cache()
{
  static const KernelCache kc = init_kernel_cache();
  return kc;
}

HyPas KernelCache::at(const CacheKey& ckey, bool swap_ab) const
{

  CacheKeyPresence ckp = check_for(ckey);
  if (!ckp.is_present)
  {
    throw miog_error("(in HyPas KernelCache::at)  " + ckp.msg);
  }
  return vals.at(ckey).get_reflected(swap_ab);
}

const HyPas& KernelCache::at(const CacheKey& ck) const
{
  CacheKeyPresence ckp = check_for(ck);
  if (!ckp.is_present)
  {
    throw miog_error("(in const HyPas & KernelCache::at)  " + ckp.msg);
  }
  return vals.at(ck);
}

void KernelCache::add(const CacheKey& ckey, const HyPas& hp)
{

  if (redirection::get_is_not_canonical(ckey.gg))
  {
    throw miog_error("internal logic error : CacheKey has geometry in non-canonical form (in add)");
  }

  CacheKeyPresence ckp = check_for(ckey);
  if (ckp.is_present)
  {

    bool              is_not_canonical = false;
    std::stringstream ss;
    ss << "Cannot add cache entry if one already exists, with. Keys: " << ckey.get_string()
       << "The existing entry is " << at(ckey, is_not_canonical).get_string()
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

std::vector<std::string> get_devices(const std::vector<CacheKey>& cks)
{
  std::vector<std::string> devices;
  for (auto& ck : cks)
  {

    if (std::find(devices.begin(), devices.end(), ck.dvc) == devices.end())
    {
      devices.push_back(ck.dvc);
    }
  }
  return devices;
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

  // TODO : distance between devices, how should this work?
  distance += 1e-6 * (dvc != ck.dvc);

  // TODO : improved distance between constraints. will be non-sym.
  distance += 1 * (constraints.get_string() != ck.constraints.get_string());

  return distance;
}
}
