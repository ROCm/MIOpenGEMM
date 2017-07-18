/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_KERNELCACHE_HPP
#define GUARD_MIOPENGEMM_KERNELCACHE_HPP

#include <miopengemm/solution.hpp>
#include <unordered_map>
#include <functional>

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
    
  std::string dvc;
  Constraints constraints;
  Geometry gg;
  std::string concatenated;  


  public:

  std::string get_concatenated() const{
    return concatenated;
  }
 
  bool operator==(const CacheKey & rhs) const{
    return concatenated == rhs.concatenated;
  }
  
  CacheKey(const std::string&, const Constraints&, const Geometry&);
  std::string get_string() const;
};

class CacheKeyHash{
  private:
  std::hash<std::string> __hash;
  
  public:
  size_t operator()(const CacheKey & ck) const;
};


class CachedSolution
{
  public:
  HyPas hp;
  SolutionStatistics stats;
  
  CachedSolution(const HyPas & hp_, SolutionStatistics stats_)
    : hp(hp_), stats(stats_)
  {
  }
  
  CachedSolution() = default;
  std::string get_string() const;
};

class KernelCache
{
  // TODO : ordered vs unordered maps
  private:
  using St = std::string;
  //std::map<St, std::unordered_map<St, std::map<St, CachedSolution>>> vals;
  std::unordered_map<CacheKey, CachedSolution, CacheKeyHash> vals;

  public:
  CacheKeyPresence check_for(const CacheKey& ck) const;
  CachedSolution at(const CacheKey& ck) const;
  void add(const CacheKey& ckey, const CachedSolution& tgcs);
};

KernelCache get_kernel_cache();

CachedSolution get_generic_cached_solution(const Constraints& constraints,
                                           const Geometry&    gg);

// [device][constraint][geometry] -> cached solution
extern const KernelCache kernel_cache;

}

#endif
