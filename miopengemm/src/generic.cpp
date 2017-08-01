#include <miopengemm/generic.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/nearest.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/bundle.hpp>
#include <miopengemm/timer.hpp>

namespace MIOpenGEMM
{

HyPas get_generic(const Geometry & gg, const Constraints& constraints){
  
  HyPas hp;
  
  
  if (gg.m >= 1000 && gg.n >= 1000){
    hp = {{"MIC5_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1", "MIC4_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1", "UNR16_GAL1_PUN0_ICE1_IWI0_SZT0_NAW64_UFO0_MAC256_SKW10_AFI1_MIA1"}};
  }
  
  else if (gg.m >= 100 && gg.n >= 100){
    hp = {{"MIC1_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1", "MIC2_PAD1_PLU0_LIW1_MIW0_WOS0_VEW1", "UNR64_GAL3_PUN1_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA1"}};
  }
  
  else{
    hp = {{"MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1", "MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1", "UNR4_GAL1_PUN0_ICE1_IWI0_SZT1_NAW64_UFO0_MAC1_SKW10_AFI0_MIA0"}};
  }
  
  hp.replace_where_defined(constraints);
  if (!Derivabilty(hp, gg).is_derivable){
    std::stringstream errm;
    errm << "Generic solution is not derivable. Consider running find for full graph search."
    << "hp (post constraint application) is " << hp.get_string() << '\n'
    << " Message was " << Derivabilty(hp, gg).msg; 
    throw miog_error(errm.str());
  }
  
  return hp;
}


Solution get_default(cl_command_queue command_queue, const Geometry & gg, const Constraints& constraints, owrite::Writer& mowri, IfNoCache::E enoc){
  oclutil::DevInfo devinfo(command_queue);
  double extime = 0;
  HyPas hp; 
  
  
  Timer timer;
  timer.start();
  
  CacheKey ck(devinfo.identifier, constraints, gg);
  Graph graph(gg, devinfo, constraints, mowri);
  
  
  if (nearest::is_within(ck, graph, kernel_cache, 0.1 * std::numeric_limits<double>::max()))
  {
    auto nearest_ck = nearest::get(ck, graph, kernel_cache);
    bool is_not_canonical = redirection::get_is_not_canonical(gg);
    hp  = kernel_cache.at(nearest_ck, is_not_canonical);

    mowri << "Nearest match in kernel cache:\n"
            << nearest_ck.get_string() << Flush;
       //<< '\n' << kernel_cache.at(nearest_ck, false).get_string() << "\n"
       //<< '\n' << kernel_cache.at(nearest_ck, true).get_string() << "\n";

  }
  

  else
  {
    if (enoc == IfNoCache::GENERIC){
      hp = get_generic(gg, constraints);
      mowri << "No kernel cache match found, returning generic.\n";
    }
    else{
      hp = graph.get_random_valid_start();
      mowri << "No kernel cache match found, returning random valid.\n";
    }
  }
  
  mowri << "Time in get_default : " << timer.get_elapsed() << Endl;
  
  kerngen::Bundle bundle(hp, gg, mowri);
  return {gg, extime, bundle.v_tgks, hp, devinfo, constraints};
}


}
