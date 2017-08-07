/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <miopengemm/nearest.hpp>

namespace MIOpenGEMM
{
namespace nearest
{

bool is_within(const CacheKey& ck, const Graph& graph, const KernelCache& kc, double threshold)
{
  for (auto& key : kc.get_keys())
  {
    if (graph.contains(kc.at(key)) && key.get_distance(ck) < threshold &&
        Derivabilty(kc.at(key), ck.gg).is_derivable)
    {
      return true;
    }
  }
  return false;
}

CacheKey get(const CacheKey& ck, const Graph& graph, const KernelCache& kc)
{

  if (!is_within(ck, graph, kc, std::numeric_limits<double>::max()))
  {
    throw miog_error("In get, with none within radius <double>::max.");
  }
  double d_nearest_derivable = std::numeric_limits<double>::max();

  auto cache_keys = kc.get_keys();
  if (cache_keys.size() == 0)
  {
    throw miog_error("No cache keys. Possibly not included in kernelcache.cpp, very strange");
  }

  CacheKey nearest_derivable(cache_keys.back());

  for (auto& key : cache_keys)
  {
    auto distance = ck.get_distance(key);
    if (distance < d_nearest_derivable)
    {
      auto hp = kc.at(key);
      if (graph.contains(kc.at(key)) && Derivabilty(hp, ck.gg).is_derivable)
      {
        d_nearest_derivable = distance;
        nearest_derivable   = key;
      }
    }
  }

  // confirm derivability
  Derivabilty drvble(kc.at(nearest_derivable), ck.gg);
  if (!drvble.is_derivable)
  {
    throw miog_error("internal logic error : nearest is not derivable. msg : " + drvble.msg);
  }
  return nearest_derivable;
}
}
}
