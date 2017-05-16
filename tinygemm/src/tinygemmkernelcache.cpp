#include <string>
#include <map>
#include <sstream>

#include <tinygemm/tinygemmkernelcache.hpp>


namespace tinygemm{

const KernelCache kernel_cache = get_kernel_cache();


KernelCache get_kernel_cache(){
KernelCache kc;

    /* There are two ways to add cache snip snips. */
    /* (1) paste them here like this : */
add_entry(kc, "some_device_key",
"some_constraint_string",
"tC0_tA0_tB0_colMaj1_m5000_n5000_k5000_lda5000_ldb5000_ldc5000_ws1_f32",
"",
{"A_MIC8_PAD2_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW1_MIW0_WOS0__C_UNR8_GAL2_PUN0_ICE1_NAW64_UFO0_MAC256_SKW10",
{59.2006, 4222.93, 3.32959, "Sun May 14 12:29:44 2017",
{3, 2, 1, Max}}});

      /* or drop them into a txt file like like this: */
#include "cacheexample.cachetxt"

return kc;
}



void add_entry(KernelCache & kc, const std::string & k_dev, const std::string & k_con,  const std::string k_geo, const std::string k_comment, tinygemm::TinygemmCachedSolution tgcs){

  
  if (kc.count(k_dev) == 0){
    kc[k_dev] = {};
  }
  
  if (kc.at(k_dev).count(k_con) == 0){
    kc[k_dev][k_con] = {};
  }
  
  if (kc.at(k_dev).at(k_con).count(k_geo) == 0){
    kc[k_dev][k_con][k_geo] = {};
  }

  if (kc.at(k_dev).at(k_con).at(k_geo).count(k_comment) == 0){
    kc[k_dev][k_con][k_geo][k_comment] = tgcs;
  }
    
  else{
    std::stringstream ss;
    ss << "An attempt to add a cache entry where one already exists, with keys \n";
    ss << get_cache_keys_string(k_dev, k_con, k_geo, k_comment);
    
    ss << "\nThe existing entry is,\n";
    ss << kc[k_dev][k_con][k_geo][k_comment].get_string() << "\n";
    ss << "\nThe proposed entry is,\n";
    ss << tgcs.get_string();
    ss << "\nPlease choose between these and remove one.\n";
    
    throw tinygemm::tinygemm_error(ss.str());
  }
}




void enforce_constraints(std::string & hps_to_update, const std::string & constraints_string, const tinygemm::TinyGemmGeometry & gg){
  
  tinygemm::openclutil::OpenCLDeviceInfo devinfo;
  hyperparams::Graph graph(gg, devinfo, hps_to_update, true);
  hyperparams::HyperParams hp(graph);
  
  auto all_constraints = hyperparams::get_all_constraints(constraints_string);
  hp.replace_where_source_defined(all_constraints);
  hps_to_update = hp.get_string();
}


TinygemmCachedSolution get_generic_cached_solution(const std::string & constraints_string, const tinygemm::TinyGemmGeometry & gg){


  /* the case where there is no cached solution */
  TinygemmCachedSolution cached_soln;
  
  if (gg.m*gg.n > 2000*2000 && gg.m >= 256 && gg.n >= 256){ /* was 5719.51 gflops on Fiji at Tue May 16 08:38:59 2017 */ 
    cached_soln = {"A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD2_PLU1_LIW1_MIW0_WOS0__C_UNR8_GAL2_PUN0_ICE1_NAW16_UFO0_MAC256_SKW10", {0,0,0, "None", {200, 10, 3, Max}}};
  }
  
  else if (gg.m*gg.n > 800*800 && gg.m >=  256. && gg.n >= 128){ /* was 3095 on a Fiji on Tue May 16 08:46:49 2017  */
    cached_soln = {"A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN1_ICE1_NAW64_UFO0_MAC256_SKW10", {0,0,0, "None", {200, 10, 3, Max}}};
  }
  
  else if (gg.m*gg.n > 300*300  && gg.m >=  64 && gg.n >= 64){
    cached_soln = {"A_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC4_PAD2_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9", {0,0,0, "None", {200, 10, 3, Max}}};
  }
  
  else if (gg.m*gg.n > 128*128 && gg.m >= 16 && gg.n >= 16){ /* Tue May 16 09:07:04 2017 */
    cached_soln = {"A_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC2_PAD2_PLU1_LIW0_MIW0_WOS0__C_UNR32_GAL2_PUN1_ICE1_NAW64_UFO0_MAC64_SKW9", {0,0,0, "None", {200, 10, 3, Max}}};    
  }
  
  else if (gg.m >= 16 && gg.n >= 16){
    cached_soln = {"A_MIC1_PAD0_PLU1_LIW0_MIW1_WOS0__B_MIC1_PAD0_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL2_PUN1_ICE1_NAW64_UFO0_MAC256_SKW10", {0,0,0, "None", {200, 10, 3, Max}}};    
 
  }
  
  else if (gg.m >= 8 && gg.n >= 8){
    cached_soln = {"A_MIC1_PAD0_PLU1_LIW0_MIW1_WOS0__B_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL3_PUN1_ICE1_NAW64_UFO0_MAC16_SKW10", {0,0,0, "None", {200, 10, 3, Max}}};     
  }

  else if (gg.m >= 4 && gg.n >= 4){
    cached_soln = {"A_MIC1_PAD2_PLU0_LIW1_MIW1_WOS0__B_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL2_PUN0_ICE1_NAW64_UFO0_MAC16_SKW10", {0,0,0, "None", {200, 10, 3, Max}}};     
  }
  
  else{
    cached_soln = {"A_MIC1_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC1_PAD2_PLU1_LIW1_MIW1_WOS0__C_UNR16_GAL2_PUN0_ICE1_NAW16_UFO0_MAC1_SKW10", {0,0,0, "None", {200, 10, 3, Max}}};         
  }
  
  enforce_constraints(cached_soln.hyperstring, constraints_string, gg);
  
  return cached_soln;

} 


std::string TinygemmCachedSolution::get_string(){
  std::stringstream ss;
  ss << "(hyperstring) " << hyperstring << "\n";
  ss << "(stats) " << stats.get_string();
  return ss.str();
}

std::string get_cache_keys_string(std::string k_dev, std::string k_con, std::string k_geo, std::string k_comment){
  std::stringstream ss;
  ss << "device key         :   `" << k_dev << "'\n";
  ss << "constraints_string :   `" << k_con << "'\n";
  ss << "geometry key       :   `" << k_geo << "'\n";
  ss << "comment key        :   `" << k_comment << "'\n";
  return ss.str();
}




}
