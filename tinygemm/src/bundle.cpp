#include <MIOpenGEMM/bundle.hpp>
#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/derivedparams.hpp>
#include <MIOpenGEMM/alphagenerator.hpp>
#include <MIOpenGEMM/copygenerator.hpp>
#include <MIOpenGEMM/normalformgenerator.hpp>
#include <MIOpenGEMM/betacgenerator.hpp>
#include <MIOpenGEMM/stringutilbase.hpp>

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

namespace MIOpenGEMM{
namespace kerngen{


class BundleGenerator{


private:
  const hyperparams::HyperParams & hp;
  const Geometry & gg;
  const derivedparams::DerivedParams dp;
  outputwriting::OutputWriter & mowri;
  bool bundle_verbose;

public: 
  BundleGenerator(const hyperparams::HyperParams & hp_, const Geometry & gg_, outputwriting::OutputWriter & mowri_, bool bundle_verbose_): hp(hp_), gg(gg_), dp(hp, gg), mowri(mowri_), bundle_verbose(bundle_verbose_) {
    
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
      throw miog_error("hp.at(nsHP::matA).vs[nsHP::WOS] should be 0, 1 or 2");
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
      throw miog_error("hp.at(nsHP::matB).vs[nsHP::WOS] should be 0, 1 or 2");
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


    if (bundle_verbose == true){
      mowri << "\n";
      mowri << "network of kernel dependencies: \n";
      for (unsigned i = 0; i < v_tgks.size(); ++i){
        std::stringstream pre_waits_for_ss;
        pre_waits_for_ss << "kernel " << i << " ( " << types[i].full << " )";
        std::string pre_waits_for = pre_waits_for_ss.str();
        mowri << pre_waits_for;
        int base_space (26);
        std::string space1 (std::max(1, base_space - static_cast<int>(pre_waits_for.size())), ' ');
        mowri << space1 << "waits for   " << Flush;
        
        if (v_wait_indices[i].size() == 0){
          mowri << "(nothing)";
        }
        
        for (unsigned j = 0; j < v_wait_indices[i].size(); ++j){
          mowri << "(kernel " << v_wait_indices[i][j] << " ( " << types[v_wait_indices[i][j]].full << " ))   " << Flush;
        }
        mowri << Endl;
      }
      mowri << "\n";

    }

    return { std::move(v_tgks), std::move(v_wait_indices), std::move(dp) };
  }
};




Bundle get_bundle(const hyperparams::HyperParams & hp,  const Geometry & gg, outputwriting::OutputWriter & mowri, bool bundle_verbose){
  
  BundleGenerator ksbg(hp, gg, mowri, bundle_verbose);
  return ksbg.generate();
}



}
}

