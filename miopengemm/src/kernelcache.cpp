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

CacheKeyPresence KernelCache::check_for(const CacheKey& ckey) const
{

  std::string open = "Failed to find cache entry from keys: " + ckey.get_string();
  std::string reason;
  std::string close = " (see examples/gencache.cpp for example generating cache entry)";

  if (vals.count(ckey.dvc) == 0)
  {
    reason = "Unrecognised device identifier in cache, maybe no cache for this device yet. ";
  }

  else if (vals.at(ckey.dvc).count(ckey.cns) == 0)
  {
    reason = "Unrecognised constraints in cache";
  }

  else if (vals.at(ckey.dvc).at(ckey.cns).count(ckey.geo) == 0)
  {
    reason = "Unrecognised geometry in cache";
  }

  else if (vals.at(ckey.dvc).at(ckey.cns).at(ckey.geo).count(ckey.cmm) == 0)
  {
    reason = "Unrecognised k_comment in cache";
  }
  else
  {
    return {};
  }
  return open + reason + close;
}

CacheKey::CacheKey(const std::string& dvc_,
                   const std::string& cns_,
                   const std::string& geo_,
                   const std::string& cmm_)
  : dvc(dvc_), cns(cns_), geo(geo_), cmm(cmm_)
{
}

std::string CacheKey::get_string() const
{
  std::stringstream ss;
  ss << "device       :   `" << dvc << "'\n";
  ss << "constraints  :   `" << cns << "'\n";
  ss << "geometry     :   `" << geo << "'\n";
  ss << "comment      :   `" << cmm << "'\n";
  return ss.str();
}

const KernelCache kernel_cache = get_kernel_cache();

KernelCache get_kernel_cache()
{
  KernelCache kc;

  // There are two ways to add cache snip snippets, the first is
  // to paste them here like this
  add_entry(
    kc,
    "some_device_key",
    "some_constraint_string",
    "tC0_tA0_tB0_colMaj1_m5000_n5000_k5000_lda5000_ldb5000_ldc5000_ws1_f32",
    "",
    {"A_MIC8_PAD2_PLU0_LIW0_MIW0_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW0_WOS0__"
     "C_UNR8_GAL2_PUN0_ICE1_NAW64_UFO0_MAC256_SKW10",
     {59.2006, 4222.93, 3.32959, "Sun May 14 12:29:44 2017", {3, 2, 1, 1e9, SummStat::E::MAX}}});

// and the second is to drop them into a txt file like like this
#include "cacheexample.cachetxt"
#include "deepbench.cachetxt"

  return kc;
}

void add_entry(KernelCache&       kc,
               const std::string& k_dev,
               const std::string& k_con,
               const std::string  k_geo,
               const std::string  k_comment,
               CachedSolution     tgcs)
{

  CacheKey ckey(k_dev, k_con, k_geo, k_comment);
  kc.add(ckey, tgcs);
}

CachedSolution KernelCache::at(const CacheKey& ckey) const
{
  CacheKeyPresence ckp = check_for(ckey);
  if (!ckp.is_present)
  {
    throw miog_error(ckp.msg);
  }
  return vals.at(ckey.dvc).at(ckey.cns).at(ckey.geo).at(ckey.cmm);
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

  if (vals.count(ckey.dvc) == 0)
  {
    vals[ckey.dvc] = {};
  }

  if (vals.at(ckey.dvc).count(ckey.cns) == 0)
  {
    vals[ckey.dvc][ckey.cns] = {};
  }

  if (vals.at(ckey.dvc).at(ckey.cns).count(ckey.geo) == 0)
  {
    vals[ckey.dvc][ckey.cns][ckey.geo] = {};
  }

  vals[ckey.dvc][ckey.cns][ckey.geo][ckey.cmm] = tgcs;
}

void enforce_constraints()
{

  // std::string&       hps_to_update,
  // const std::string& constraints_string,
  // const Geometry&    gg)

  throw miog_error("enforce_constraints in kernelcache not implememented");
  // oclutil::DevInfo devinfo;
  // Graph           graph(gg, devinfo, constraints_string);
  // HyPas     hp();

  // auto all_constraints = get_all_constraints(constraints_string);
  // hp.replace_where_source_defined(all_constraints);
  // hps_to_update = hp.get_string();
}

/* TODO : certain of these kernels are slow,
 * so as to cover the case (ROCm) of
 * problems in compilation
 * with small k. Fix this */
CachedSolution get_generic_cached_solution(const std::string& constraints_string,
                                           const Geometry&    gg)
{

  // the case where there is no cached solution
  CachedSolution cached_soln;

  if (gg.m * gg.n > 2000 * 2000 && gg.m >= 256 && gg.n >= 256)
  {
    cached_soln = {"A_MIC8_PAD1_PLU0_LIW0_MIW0_WOS0__B_MIC6_PAD2_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR8_GAL2_"
                   "PUN0_ICE1_NAW16_UFO0_MAC256_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  else if (gg.m * gg.n > 800 * 800 && gg.m >= 256. && gg.n >= 128)
  {
    cached_soln = {"A_MIC8_PAD1_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR16_GAL1_"
                   "PUN1_ICE1_NAW64_UFO0_MAC256_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  else if (gg.m * gg.n > 300 * 300 && gg.m >= 64 && gg.n >= 64)
  {
    cached_soln = {"A_MIC2_PAD1_PLU0_LIW0_MIW0_WOS0__B_MIC2_PAD2_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR16_GAL3_"
                   "PUN0_ICE1_NAW64_UFO0_MAC256_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  else if (gg.m * gg.n > 128 * 128 && gg.m >= 16 && gg.n >= 16)
  {
    cached_soln = {"A_MIC1_PAD1_PLU0_LIW0_MIW0_WOS0__B_MIC2_PAD2_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR32_GAL2_"
                   "PUN1_ICE1_NAW64_UFO0_MAC64_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  else if (gg.m >= 16 && gg.n >= 16)
  {
    cached_soln = {"A_MIC1_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC1_PAD0_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR16_GAL2_"
                   "PUN1_ICE1_NAW64_UFO0_MAC256_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  else if (gg.m >= 8 && gg.n >= 8)
  {
    cached_soln = {"A_MIC1_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC2_PAD1_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR16_GAL3_"
                   "PUN1_ICE1_NAW64_UFO0_MAC16_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  else if (gg.m >= 4 && gg.n >= 4)
  {
    cached_soln = {"A_MIC1_PAD2_PLU0_LIW0_MIW0_WOS0__B_MIC1_PAD1_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR16_GAL2_"
                   "PUN0_ICE1_NAW64_UFO0_MAC16_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  else
  {
    cached_soln = {"A_MIC1_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC1_PAD0_PLU0_LIW0_"
                   "MIW0_WOS0__C_UNR8_GAL1_"
                   "PUN0_ICE1_NAW16_UFO0_MAC1_SKW10",
                   {0, 0, 0, "None", {200, 10, 3, 1e12, SummStat::MAX}}};
  }

  (void)constraints_string;
  throw miog_error("get_generic_cached_solution not impled completeky");
  // enforce_constraints(cached_soln.hyperstring, constraints_string, gg);

  return cached_soln;
}

std::string CachedSolution::get_string() const
{
  std::stringstream ss;
  ss << "(hyperstring) " << hyperstring << "\n";
  ss << "(stats) " << stats.get_string();
  return ss.str();
}
}
