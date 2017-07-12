/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <miopengemm/architests.hpp>
#include <miopengemm/bundle.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>
#include <miopengemm/jinx.hpp>

namespace MIOpenGEMM
{




cl_mem get_single(cl_command_queue command_queue, const std::string& hash)
{
  size_t c_memsize = 1;
  cl_mem single;
  oclutil::cl_set_buffer_from_command_queue(
    single,
    command_queue,
    CL_MEM_READ_WRITE,
    c_memsize,
    NULL,
    hash + ", in function cl_mem get_single which returns a cl_mem",
    true);
  return single;
}

Solution find(cl_command_queue             command_queue,
              const FindParams&            find_params,
              cl_mem                       a,
              cl_mem                       b,
              cl_mem                       c,
              cl_mem                       workspace,
              const std::string            constraints_string,
              const Geometry&              gg,
              const Offsets&               toff,
              owrite::Writer&              mowri,
              bool                         c_is_const)
{

  gg.check_ldx_consistent();
  bool full_constraints_expected = false;

  Jinx oger(                  command_queue,
                              gg,
                              toff,
                              a,
                              b,
                              c,
                              c_is_const,
                              workspace,
                              constraints_string,
                              full_constraints_expected,
                              mowri);
                              
  return oger.find(find_params);
}

std::tuple<bool, std::string> check_for_default(cl_command_queue command_queue,
                                                std::string     constraints_string,
                                                const Geometry& gg,
                                                std::string     k_comment)
{

  oclutil::DevInfo devinfo(command_queue);
  std::string                  k_dev = devinfo.identifier;
  std::string                  k_con = constraints_string;
  std::string                  k_geo = gg.get_string();

  std::stringstream ss;
  ss << "\nfailed to find cache entry from keys:\n";
  ss << get_cache_keys_string(k_dev, k_con, k_geo, k_comment);

  std::string final_comment(
    "(see tests/gencache.cpp for an example of generating a cache entry)\n");

  if (kernel_cache.count(k_dev) == 0)
  {
    ss << "Unrecognised device identifier in cache.\nMaybe the cache needs to "
          "be built for this "
          "device? \n"
       << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).count(k_con) == 0)
  {
    ss << "Unrecognised constraints_string in cache.\nMaybe the cache needs to "
          "be built with these "
          "constraints? \n"
       << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).at(k_con).count(k_geo) == 0)
  {
    ss << "Unrecognised geometry key (gg.get_string()) in cache.\nMaybe a "
          "cache entry needs to be "
          "generated with this geometry? \n"
       << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).at(k_con).at(k_geo).count(k_comment) == 0)
  {
    ss << "Unrecognised k_comment in cache\n";
    return std::make_tuple(false, ss.str());
  }

  return std::make_tuple(true, "");
}

// fall back solution
Solution get_default(const Geometry& gg)
{
  std::string constraints_string = "";

  auto                         cached_soln = get_generic_cached_solution(constraints_string, gg);
  oclutil::DevInfo devinfo;
  owrite::Writer  mowri(Ver::E::SILENT, "");
  Graph           graph(gg, devinfo, cached_soln.hyperstring, false);
  HyperParams     hp(graph);

  bool bundle_verbose_get_default = true;
  auto bundle                     = kerngen::get_bundle(hp, gg, mowri, bundle_verbose_get_default);

  return {gg, cached_soln.stats, bundle.v_tgks, hp.get_string(), devinfo, constraints_string};
}

Solution get_default(cl_command_queue             command_queue,
                     std::string                  constraints_string,
                     const Geometry&              gg,
                     std::string                  k_comment,
                     owrite::Writer& mowri)
{

  oclutil::DevInfo devinfo(command_queue);

  std::string k_dev = devinfo.identifier;
  std::string k_con = constraints_string;
  std::string k_geo = gg.get_string();

  CachedSolution cached_soln;
  auto           pair = check_for_default(command_queue, constraints_string, gg, k_comment);
  if (std::get<0>(pair) == false)
  {
    miog_warning(std::get<1>(pair));
    mowri << std::get<1>(pair);
    cached_soln = get_generic_cached_solution(constraints_string, gg);
  }

  else
  {
    cached_soln = kernel_cache.at(k_dev).at(k_con).at(k_geo).at(k_comment);
  }

  // generating source files from cache
  Graph       graph(gg, devinfo, cached_soln.hyperstring, false);
  HyperParams hp(graph);
  bool                     bundle_verbose_get_default = true;
  auto                     bundle = kerngen::get_bundle(hp, gg, mowri, bundle_verbose_get_default);

  return {gg, cached_soln.stats, bundle.v_tgks, hp.get_string(), devinfo, constraints_string};
}

void benchgemm(cl_command_queue             command_queue,
               const std::string&           hyperstring,
               size_t                     max_n_runs,
               double                     max_time,
               const Geometry&              gg,
               const Offsets&               toff,
               cl_mem                       a_gpu,
               cl_mem                       b_gpu,
               cl_mem                       c_gpu,
               cl_mem                       workspace_gpu,
               owrite::Writer& mowri,
               bool                         c_is_const)
{

  bool full_constraints_expected = true;

  Jinx oger(command_queue,
                              gg,
                              toff,
                              a_gpu,
                              b_gpu,
                              c_gpu,
                              c_is_const,
                              workspace_gpu,
                              hyperstring,
                              full_constraints_expected,
                              mowri);

  oger.benchgemm(max_n_runs, max_time);
  //}
}

Solution find(float            allotted_time,
              cl_command_queue command_queue,
              cl_mem           a,
              cl_mem           b,
              cl_mem           c,
              bool             enforce_determinism,
              const Geometry&  tgg,
              bool             with_warnings)
{

  Solution solution = get_default(tgg);

  /* TODO : where is a good place to set this ? */
  float min_time_without_cache = 100.00;

  SummStat::E sumstat(SummStat::E::MEDIAN);
  size_t    allotted_descents = 30;
  size_t    max_n_runs_per_kernel = 3;
  double    max_time_per_kernel = 1000.; // 1000 seconds. 
  
  FindParams  find_params(allotted_time, allotted_descents, max_n_runs_per_kernel, max_time_per_kernel, sumstat);

  cl_mem workspace = nullptr;

  std::string constraints_string = "A_WOS0__B_WOS0"; // no workspace
  if (enforce_determinism == true)
  {
    constraints_string += "__C_ICE1";
  }

  Offsets toff(0, 0, 0, 0, 0, 0, 0, 0);

  owrite::Writer mowri(Ver::E::TERMINAL, "");

  bool c_is_const = true;

  std::string k_comment = "";
  auto        pair      = check_for_default(command_queue, constraints_string, tgg, k_comment);

  bool is_custom_cache_entry = std::get<0>(pair);

  mowri << "is_custom_cache_entry = " << is_custom_cache_entry << Endl;
  if (allotted_time < min_time_without_cache)
  {
    mowri << "allotted_time < min_time_without_cache, will not search\n";

    if (is_custom_cache_entry == false)
    {

      std::stringstream ss;
      ss << "\n\n"
         << "In find (version without workspace), and "
         << "\n(1) allotted_time (" << allotted_time << ") is less than min_time_without_cache ("
         << min_time_without_cache << ")  "
         << "\n(2) there is no custom cache entry. The message returned when "
         << "attempting to obtain "
         << "a custom cache entry was,"
         << '\n'
         << std::get<1>(pair) << '\n'
         << "Either "
         << "\n(1) set allotted_time to be greater than "
         << "min_time_without_cache, or "
         << "\n(2) generate a custom cache entry (see tests/gencache.cpp for "
         << "an example)."
         << "\n\nReturing a generic cache entry\n";

      mowri << ss.str();

      if (with_warnings)
        miog_warning("\nvery limited search with no custom cache : expect a "
                     "sub-optimal kernel(s) \n");

      solution = get_default(tgg);
    }

    else
    {
      solution = get_default(command_queue, constraints_string, tgg, k_comment, mowri);
    }
  }

  // we have time to search
  else
  {
    auto found_soln = find(command_queue,
                           find_params,
                           a,
                           b,
                           c,
                           workspace,
                           constraints_string,
                           tgg,
                           toff,
                           mowri,
                           c_is_const);

    if (std::get<0>(pair) == false)
    {
      solution = found_soln;
    }

    else
    {
      auto cached_soln = get_default(command_queue, constraints_string, tgg, k_comment, mowri);
      if (cached_soln.statistics.median_benchmark_gflops >
          found_soln.statistics.median_benchmark_gflops)
      {
        mowri << "cached solution has better glops: "
              << cached_soln.statistics.median_benchmark_gflops << ", returning cached soln"
              << Endl;
        solution = cached_soln;
      }
      else
      {
        mowri << "cached solution has worse glops: "
              << cached_soln.statistics.median_benchmark_gflops
              << ", the new soln will be returned\n"
              << "consider adding this new solution to the cache, it's entry "
              << "string is\n"
              << cached_soln.get_cache_entry_string() << Endl;
        solution = found_soln;
      }
    }
  }

  return solution;
}

}

