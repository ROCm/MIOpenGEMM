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
add_entry(kc, "FijiOpenCL1p2AMDAPP2264p102264p10", "", "tC0_tA0_tB0_colMaj1_m500_n500_k256_lda500_ldb256_ldc500_ws10_f32", "", {"kernelstring", "kyoolbananas"});
return kc;
}

}
