/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNELCACHE_HPP
#define GUARD_MIOPENGEMM_KERNELCACHE_HPP

#include <miopengemm/solution.hpp>

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

  public:
  std::string dvc;  // device
  std::string cns;  // constraint
  std::string geo;  // geometry
  std::string cmm;  // comment
  CacheKey(const std::string&, const std::string&, const std::string&, const std::string&);
  std::string get_string() const;
};

class CachedSolution
{
  public:
  std::string        hyperstring;
  SolutionStatistics stats;
  CachedSolution(std::string hyperstring_, SolutionStatistics stats_)
    : hyperstring(hyperstring_), stats(stats_)
  {
  }
  CachedSolution() = default;

  std::string get_string() const;
};

class KernelCache
{
  /* TODO : unordered maps are faster */
  private:
  using St = std::string;
  std::map<St, std::map<St, std::map<St, std::map<St, CachedSolution>>>> vals;

  public:
  CacheKeyPresence check_for(const CacheKey& ck) const;
  CachedSolution at(const CacheKey& ck) const;
  void add(const CacheKey& ckey, const CachedSolution& tgcs);
};

KernelCache get_kernel_cache();

CachedSolution get_generic_cached_solution(const std::string& constraints_string,
                                           const Geometry&    gg);

// [device][constraint][geometry][further_comment] -> cached solution
extern const KernelCache kernel_cache;

void add_entry(KernelCache&       kc,
               const std::string& k_dev,
               const std::string& k_con,
               const std::string  k_geo,
               const std::string  k_comment,
               CachedSolution     tgcs);
}

#endif
