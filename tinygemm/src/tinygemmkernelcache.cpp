#include <string>
#include <map>
#include <sstream>

#include <tinygemm/tinygemmkernelcache.hpp>


namespace tinygemm{


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

const KernelCache kernel_cache = get_kernel_cache();


KernelCache get_kernel_cache(){
KernelCache kc;

      /* add from snip snip here like so: */

add_entry(kc, "FijiOpenCL1p2AMDAPP2264p102264p10", /* device key */
"A_WOS0__B_WOS0", /* constraint key */
"tC0_tA0_tB0_colMaj1_m5000_n5000_k5000_lda5000_ldb5000_ldc5000_ws1_f32", /* geometry key */
"", /* geometry key */
{"A_MIC8_PAD2_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW1_MIW0_WOS0__C_UNR8_GAL2_PUN0_ICE1_NAW64_UFO0_MAC256_SKW10", /* solution hyper string */
{59.2006, 4222.93, 3.32959, "Sun May 14 12:29:44 2017", /* solution stats (time [ms], gflops, time found (within descent), date found */
{3, 2, 1, Max}}}); /* find param: allotted time, allotted descents, n runs per kernel, summmary over runs */




add_entry(kc, "FijiOpenCL1p2AMDAPP2264p102264p10", /* device key */
"A_WOS0__B_WOS0", /* constraint key */
"tC0_tA0_tB0_colMaj1_m16_n64_k800_lda16_ldb800_ldc16_ws0_f32", /* geometry key */
"", /* comment key */
{"A_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MIC1_PAD1_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN1_ICE2_NAW16_UFO0_MAC256_SKW11", /* solution hyper string */
{0.24576, 6.66667, 0.952173, "Mon May 15 15:18:29 2017", /* solution stats (time [ms], gflops, time found (within descent), date found */
{1, 100, 3, Max}}}); /* find param: allotted time, allotted descents, n runs per kernel, summmary over runs */


add_entry(kc, "FijiOpenCL1p2AMDAPP2264p102264p10", /* device key */
"A_WOS0__B_WOS0", /* constraint key */
"tC0_tA1_tB0_colMaj1_m800_n64_k16_lda16_ldb16_ldc800_ws0_f32", /* geometry key */
"", /* comment key */
{"A_MIC6_PAD2_PLU0_LIW1_MIW1_WOS0__B_MIC1_PAD2_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL2_PUN0_ICE1_NAW16_UFO0_MAC256_SKW10", /* solution hyper string */
{0.04656, 35.189, 0.670019, "Mon May 15 15:22:25 2017", /* solution stats (time [ms], gflops, time found (within descent), date found */
{1, 100, 3, Max}}}); /* find param: allotted time, allotted descents, n runs per kernel, summmary over runs */





      /* or drop them in like so: */
#include "cacheexample.cachetxt"


return kc;
}

}
