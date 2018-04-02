/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <miopengemm/bundle.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/nearest.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/timer.hpp>
#include <miopengemm/tinyzero.hpp>

namespace MIOpenGEMM
{

HyPas get_generic(const Geometry& gg, const Constraints& constraints)
{

  HyPas hp;

  if (gg.m >= 1000 && gg.n >= 1000)
  {
    hp = {{{"MIC5_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1",
            "MIC4_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1",
            "UNR16_GAL1_PUN0_ICE1_IWI0_SZT0_NAW64_UFO0_MAC256_SKW10_AFI1_MIA1_MAD0"}}};
  }

  else if (gg.m >= 100 && gg.n >= 100)
  {
    hp = {{{"MIC1_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1",
            "MIC2_PAD1_PLU0_LIW1_MIW0_WOS0_VEW1",
            "UNR64_GAL3_PUN1_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA1_MAD0"}}};
  }

  else
  {
    hp = {{{"MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1",
            "MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1",
            "UNR4_GAL1_PUN0_ICE1_IWI0_SZT1_NAW64_UFO0_MAC1_SKW10_AFI0_MIA0_MAD0"}}};
  }

  hp.replace_where_defined(constraints);
  if (!Derivabilty(hp, gg).is_derivable)
  {
    std::stringstream errm;
    errm << "Generic solution is not derivable. Consider running find for full graph search."
         << "hp (post constraint application) is " << hp.get_string() << '\n'
         << " Message was " << Derivabilty(hp, gg).msg;
    throw miog_error(errm.str());
  }

  return hp;
}

Solution get_default_soln(const oclutil::DevInfo& devinfo,
                          const Geometry&         gg,
                          const Constraints&      constraints,
                          owrite::Writer&         mowri,
                          IfNoCache::E            enoc,
                          size_t                  rank)
{

  double extime = 0;
  HyPas  hp;

  auto&& kernel_cache = get_kernel_cache();

  Timer timer;
  timer.start();

  CacheKey ck(devinfo.identifier, constraints, gg);
  Graph    graph(gg, devinfo, constraints, mowri);

  bool   catch_ROCm_small_k = false;
  size_t ROCm_small_k       = 1;

  // TODO : check this.
  if ((catch_ROCm_small_k == false || gg.k > ROCm_small_k) &&
      (nearest::is_within(ck, graph, kernel_cache, 0.1 * std::numeric_limits<double>::max(), rank)))
  {
    auto nearest_ck       = nearest::get(ck, graph, kernel_cache, rank);
    bool is_not_canonical = redirection::get_is_not_canonical(gg);
    hp                    = kernel_cache.at(nearest_ck, is_not_canonical);

    mowri << "Nearest match in kernel cache:\n" << nearest_ck.get_string() << Flush;
  }

  else
  {
    if (enoc == IfNoCache::GENERIC)
    {
      hp = get_generic(gg, constraints);
      mowri << "No kernel cache match found, returning generic.\n";
    }
    else
    {
      hp = graph.get_random_valid_start();
      mowri << "No kernel cache match found, returning random valid.\n";
    }
  }

  mowri << "Time in get_default : " << timer.get_elapsed() << " [s]" << Endl;

  kerngen::Bundle bundle(hp, gg);  //, mowri);

  return {gg, extime, bundle.v_tgks, hp, devinfo, constraints};
}

Solution find(float            allotted_time,
              cl_command_queue command_queue,
              cl_mem           a,
              cl_mem           b,
              cl_mem           c,
              bool             enforce_determinism,
              const Geometry&  tgg,
              bool             verbose,
              bool             with_warnings)
{

  (void)with_warnings;
  bool   c_is_const    = true;
  cl_mem workspace_gpu = nullptr;

  Ver::E         e_ver              = verbose ? Ver::E::TERMINAL : Ver::E::SILENT;
  std::string    constraints_string = enforce_determinism ? "C__ICE1" : "";
  Constraints    constraints(constraints_string);
  auto           find_params = get_at_least_n_seconds(static_cast<double>(allotted_time));
  owrite::Writer mowri(e_ver, "");
  Offsets        offsets = get_zero_offsets();
  TinyZero       jinx(command_queue, tgg, offsets, a, b, c, c_is_const, workspace_gpu, mowri);

  size_t           rank = 0;
  oclutil::DevInfo devinfo(command_queue);
  Solution soln = get_default_soln(devinfo, tgg, constraints, mowri, IfNoCache::E::GENERIC, rank);
  if (allotted_time > 0.1f)
  {
    soln = jinx.find0(constraints, find_params);
  }

  return soln;
}

Solution get_default(const Geometry& gg)
{

  Constraints constraints{""};
  // auto           devinfo = oclutil::get_fiji_devinfo();
  auto           devinfo = oclutil::get_vega_devinfo();
  owrite::Writer mowri(Ver::E::SILENT, "");
  size_t         rank = 0;
  return get_default_soln(devinfo, gg, constraints, mowri, IfNoCache::GENERIC, rank);
}
}
