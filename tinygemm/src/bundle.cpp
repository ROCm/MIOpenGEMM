#include <tinygemm/bundle.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/betackernelutil.hpp>
#include <tinygemm/alphagenerator.hpp>

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



void indentify(std::string & source){
  std::string newsource;
  newsource.reserve(source.length());
  std::string::size_type last_lend = source.find("\n", 0);  
  std::string::size_type next_lend = source.find("\n", last_lend + 1);
  std::string::size_type next_open = source.find("{", 0);
  std::string::size_type next_close = source.find("}", 0);  
  newsource.append(source, 0, last_lend);
  int indent_level = 0;

  while (std::string::npos != next_lend){

    if (next_open < last_lend){
      indent_level += 1;
      next_open = source.find("{", next_open + 1);
    }
    else if (next_close < next_lend){
      indent_level -= 1;
      next_close = source.find("}", next_close + 1);
    }
    
    else{
      newsource.append("\n");
      for (int i = 0; i < indent_level; ++i){
        newsource.append("  ");
      }
      newsource.append(source, last_lend + 1, next_lend - last_lend - 1);
      last_lend = next_lend;
      next_lend = source.find("\n", next_lend + 1);
    }
  }
  
  newsource += source.substr(last_lend);
  source.swap(newsource);
}







class BundleGenerator{

private:
  const hyperparams::HyperParams & hp;
  const tinygemm::TinyGemmGeometry & gg;
  /* to be set in constructor based on parameters provided */
  const derivedparams::DerivedParams dp;

public: 
  BundleGenerator(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_): hp(hp_), gg(gg_), dp(hp, gg) {}

Bundle generate(){
  std::vector<KernelString> tgks;  
  
  
  if (dp.does_beta_c_inc == 0){
    tgks.emplace_back(  "betac", betac::get_betac_kernel_string(gg.floattype, tinygemm::betac::genericbetackernelname), tinygemm::betac::genericbetackernelname );
  }
  
  //AlphaGenerator ag(hp, gg, dp);

  tgks.emplace_back( alphagen::get_alpha_kernelstring(hp, gg, dp) );
  for (auto & x : tgks){
    indentify(x.kernstr);
  }
  return { std::move(tgks), std::move(dp) };
}
};




Bundle get_bundle(const hyperparams::HyperParams & hp,  const tinygemm::TinyGemmGeometry & gg){  
  BundleGenerator ksbg(hp, gg);
  return ksbg.generate();
}



}
}
