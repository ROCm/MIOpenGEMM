#include <tinygemm/bundle.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/betagenerator.hpp>
#include <tinygemm/alphagenerator.hpp>
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

namespace tinygemm{
namespace kerngen{

class BundleGenerator{

private:
  const hyperparams::HyperParams & hp;
  const tinygemm::TinyGemmGeometry & gg;
  /* to be set in constructor based on parameters provided */
  const derivedparams::DerivedParams dp;

public: 
  BundleGenerator(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_): hp(hp_), gg(gg_), dp(hp, gg) {}

Bundle generate(){
  std::vector<KernelString> v_tgks;  
  std::vector<std::vector<unsigned> > v_wait_indices;
  
  if (dp.does_beta_c_inc == 0){
    /* currently the betac kernel does not rely on any hyper parameters, this may change. */
    v_tgks.emplace_back( betagen::get_beta_kernelstring(gg) );
    v_tgks.emplace_back( alphagen::get_alpha_kernelstring(hp, gg, dp) );
    v_wait_indices = { {}, {0} };
  }
  
  else{
    v_tgks.emplace_back( alphagen::get_alpha_kernelstring(hp, gg, dp) );
    v_wait_indices = { {} };
  }
  
  for (auto & x : v_tgks){
    stringutil::indentify(x.kernstr);
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
