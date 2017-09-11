/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <algorithm>
#include <miopengemm/nearest.hpp>

namespace MIOpenGEMM
{
namespace nearest
{

bool is_within(
  const CacheKey& ck, const Graph& graph, const KernelCache& kc, double threshold, size_t rank)
{
  size_t count = 0;
  for (auto& key : kc.get_keys())
  {
    if (graph.contains(kc.at(key)) && key.get_distance(ck) < threshold &&
        Derivabilty(kc.at(key), ck.gg).is_derivable)
    {
      ++count;
      if (count > rank)
      {
        return true;
      }
    }
  }
  return false;
}

// rank = 0 for nearest, 1 for second nearest etc.
CacheKey get(const CacheKey& ck, const Graph& graph, const KernelCache& kc, size_t rank)
{
  
  //std::cout << "\n --  --  in get " << ck.gg.get_string() << std::endl;

  if (!is_within(ck, graph, kc, std::numeric_limits<double>::max(), rank))
  {
    throw miog_error("In get, with none within radius <double>::max.");
  }

  auto cache_keys = kc.get_keys();
  if (cache_keys.size() == 0)
  {
    throw miog_error("No cache keys. Possibly not included in kernelcache.cpp, very strange");
  }

  using dst_tup = std::tuple<double, size_t>;
  std::vector<dst_tup> v_di;

  for (size_t keyi = 0; keyi < cache_keys.size(); ++keyi)
  {
    auto key      = cache_keys[keyi];
    auto distance = ck.get_distance(key);

    //if (key.gg.get_string() == "tC0_tA0_tB0_colMaj1_m30_n30_k30_lda30_ldb30_ldc30_ws0_f32"){
      //std::cout << "ooooooooooooooooooooooooooooooooooooo x x x x \n asd " << std::endl;
    //}

    auto hp = kc.at(key);
    
    //if (key.gg.get_string() == "tC0_tA0_tB0_colMaj1_m30_n30_k30_lda30_ldb30_ldc30_ws0_f32"){
      //std::cout << "  00 " << Derivabilty(hp, ck.gg).is_derivable << std::endl;
      //std::cout << ck.gg.get_string() << std::endl;
      //std::cout << hp.get_string() << std::endl;
    //}    

    if (graph.contains(kc.at(key)) && Derivabilty(hp, ck.gg).is_derivable)
    {
      //if (key.gg.get_string() == "tC0_tA0_tB0_colMaj1_m30_n30_k30_lda30_ldb30_ldc30_ws0_f32"){
        //std::cout << "\n\n what ? \n\n" << std::endl;
      //}
   
      v_di.emplace_back(std::make_tuple(distance, keyi));
    }
  }

  if (rank >= v_di.size())
  {
    throw miog_error("rank too large in get, too few candidates. Use is_within to check");
  }

  std::nth_element(
    v_di.begin(), v_di.begin() + rank, v_di.end(), [](const dst_tup& a, const dst_tup& b) {
      return std::get<0>(a) < std::get<0>(b);
    });

  auto nearest_derivable = cache_keys[std::get<1>(v_di[rank])];

  // confirm derivability
  Derivabilty drvble(kc.at(nearest_derivable), ck.gg);
  if (!drvble.is_derivable)
  {
    throw miog_error("internal logic error : nearest is not derivable. msg : " + drvble.msg);
  }

  std::cout << "\nfinally 1) " << kc.at(nearest_derivable).get_string() << std::endl;
  std::cout << "\nfinally 2) " << kc.at(nearest_derivable).get_reflected(true).get_string() << std::endl;
  
  return nearest_derivable;
}
}
}
