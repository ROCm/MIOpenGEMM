#include <tinygemm/bundle.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/alphagenerator.hpp>
#include <tinygemm/copygenerator.hpp>
#include <tinygemm/normalformgenerator.hpp>
#include <tinygemm/betacgenerator.hpp>
#include <tinygemm/stringutilbase.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <tuple>
#include <fstream>
#include <chrono>
#include <ctime>
#include <algorithm>

namespace tinygemm{
namespace kerngen{


class BundleGenerator{


private:
  const hyperparams::HyperParams & hp;
  const tinygemm::TinyGemmGeometry & gg;
  /* to be set in constructor, based on parameters provided */
  const derivedparams::DerivedParams dp;
  

public: 
  BundleGenerator(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_): hp(hp_), gg(gg_), dp(hp, gg) {
    
  }

  Bundle generate(){
    
    
    std::vector<KernelString> v_tgks;  
    std::vector<std::vector<unsigned> > v_wait_indices;
    
    if (hp.at(nsHP::matA).vs[nsHP::WOS] == 0){
      //no wsa kernel
    }
    
    else if (hp.at(nsHP::matA).vs[nsHP::WOS] == 1){
      v_tgks.emplace_back( copygen::get_copya_kernelstring(hp, gg, dp) );
    }
    
    else if (hp.at(nsHP::matA).vs[nsHP::WOS] == 2){
      v_tgks.emplace_back( nformgen::get_nforma_kernelstring(hp, gg, dp) );
    }
    
    else{
      throw tinygemm_error("hp.at(nsHP::matA).vs[nsHP::WOS] should be 0, 1 or 2");
    }
    

    if (hp.at(nsHP::matB).vs[nsHP::WOS] == 0){
      //no wsb kernel
    }
    
    
    else if (hp.at(nsHP::matB).vs[nsHP::WOS] == 1){
      v_tgks.emplace_back( copygen::get_copyb_kernelstring(hp, gg, dp) ); //deduce from hp whether a is copied or not. 
    }

    else if (hp.at(nsHP::matB).vs[nsHP::WOS] == 2){
      v_tgks.emplace_back( nformgen::get_nformb_kernelstring(hp, gg, dp) );
    }

    else {
      throw tinygemm_error("hp.at(nsHP::matB).vs[nsHP::WOS] should be 0, 1 or 2");
    }

    
    if (dp.main_does_beta_c_inc == 0){
      v_tgks.emplace_back( betacgen::get_betac_kernelstring(hp, gg, dp) );
    }
    
    
    v_tgks.emplace_back( alphagen::get_alpha_kernelstring(hp, gg, dp) );

    /* indent the kernel strings, in case someone wants to print them. For (xx-minorly) better performance, this should not be done */
    for (auto & x : v_tgks){
      stringutil::indentify(x.kernstr);
    }

    std::vector<KernelType> types;
    for (unsigned i = 0; i < v_tgks.size(); ++i){
      types.push_back(v_tgks[i].type);
    }

    for (unsigned i = 0; i < v_tgks.size(); ++i){
      v_wait_indices.push_back({});
      for (unsigned j = 0; j < v_tgks.size(); ++j){
        if (std::find(kernel_dependencies.at(types[i].basic_kernel_type).begin(), kernel_dependencies.at(types[i].basic_kernel_type).end(), types[j].basic_kernel_type) != kernel_dependencies.at(types[i].basic_kernel_type).end()) {
          v_wait_indices.back().push_back(j);
        }
      }
    }


    if (true == false){
      std::cout << "------------ network ------------------- \n";
      for (unsigned i = 0; i < v_tgks.size(); ++i){
        std::cout << "------------ kernel " << i << " ( " << types[i].full << " )  ----- waits for -----> " << std::flush;
        for (unsigned j = 0; j < v_wait_indices[i].size(); ++j){
          std::cout << "------------ " << v_wait_indices[i][j] << " ( " << types[v_wait_indices[i][j]].full << " )   " << std::flush;
        }
        std::cout << std::endl;
      }
    }

    return { std::move(v_tgks), std::move(v_wait_indices), std::move(dp) };
  }
};




Bundle get_bundle(const hyperparams::HyperParams & hp,  const tinygemm::TinyGemmGeometry & gg){
  
  BundleGenerator ksbg(hp, gg);
  return ksbg.generate();
}



}
}

