#include <string>
#include <map>

#include <tinygemm/tinygemmkernelcache.hpp>


namespace tinygemm{

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
    /* TODO duplicate entries : handle */
  }
}

const KernelCache kernel_cache = get_kernel_cache();


KernelCache get_kernel_cache(){
KernelCache kc;

add_entry(kc, "FijiOpenCL1p2AMDAPP2264p102264p10",  "A_WOS0__B_WOS0",  "tC0_tA0_tB0_colMaj1_m5124_n9124_k1760_lda5124_ldb1760_ldc5124_ws1_f32",  "", {"A_MIC8_PAD2_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD2_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL2_PUN1_ICE1_NAW64_UFO0_MAC256_SKW10", "runtime:41.415  gflops:3973.55  date:Thu May 11 16:25:59 2017"});


return kc;
}

}
