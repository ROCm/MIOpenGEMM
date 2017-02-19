#ifndef BETACGENERATOR_HPP
#define BETACGENERATOR_HPP

#include <tinygemm/bylinegenerator.hpp>

namespace tinygemm{
namespace betacgen{


class BetacGenerator : public bylinegen::ByLineGenerator {

public:

BetacGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_);


  virtual void setup_additional() final override;
  
    
  
};



KernelString get_betac_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);


  
}
}

#endif


