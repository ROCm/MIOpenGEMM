/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNELCACHE_HPP
#define GUARD_MIOPENGEMM_KERNELCACHE_HPP

#include <functional>
#include <unordered_map>
#include <miopengemm/derivedparams.hpp>

namespace MIOpenGEMM
{

class CacheKeyPresence
{
  public:
  bool        is_present;
  std::string msg;
  CacheKeyPresence() : is_present(true), msg("") {}
  CacheKeyPresence(const std::string& msg_) : is_present(false), msg(msg_) {}
};

class CacheKey
{
  private:
  bool from_non_canonical;

  public:
  std::string dvc;
  // always in canonical form
  Constraints constraints;
  // always in canonical form
  Geometry    gg;
  std::string concatenated;

  bool operator==(const CacheKey& rhs) const { return concatenated == rhs.concatenated; }

  CacheKey(const std::string& device, const Constraints&, const Geometry&);
  std::string get_string() const;
  double get_distance(const CacheKey& ck) const;
};

class CacheKeyHash
{
  private:
  std::hash<std::string> __hash;

  public:
  size_t operator()(const CacheKey& ck) const;
};

class KernelCache
{
  private:
  std::unordered_map<CacheKey, HyPas, CacheKeyHash> vals;

  public:
  CacheKeyPresence check_for(const CacheKey& ck) const;
  HyPas at(const CacheKey& ck, bool swap_ab) const;
  const HyPas& at(const CacheKey& ck) const;

  // hp must be transformed if geometry is.
  void add(const CacheKey& ckey, const HyPas& hp);
  std::vector<CacheKey> get_keys() const;

  std::string get_cache_entry_string(const CacheKey& ck) const;
};

void filter_device(std::vector<CacheKey>&, const std::vector<std::string>& device_frags);
void filter_geometries(std::vector<CacheKey>&, const std::vector<Geometry>& geometries);
void filter_floattype(std::vector<CacheKey>&, size_t);

const KernelCache& get_kernel_cache();

std::string get_cache_entry_string(const CacheKey& ck, const HyPas& hypas, bool swap_ab);
std::vector<Geometry> get_geometries(const std::vector<CacheKey>& cks);
std::vector<std::string> get_devices(const std::vector<CacheKey>& cks);
}

#endif
