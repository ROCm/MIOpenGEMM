#ifndef BETACGENERATOR_HPP
#define BETACGENERATOR_HPP

#include <tinygemm/bylinegenerator.hpp>
#include <sstream>

namespace tinygemm{
namespace betacgen{


class BetacGenerator : public bylinegen::ByLineGenerator {

public:

BetacGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_);


  virtual void setup_additional() override final ;

  virtual void append_derived_definitions_additional(std::stringstream & ss) override final;

  size_t get_local_work_size() override final;    
  
  size_t get_work_per_thread() override final;
  
  
};



KernelString get_betac_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);


  
}
}

#endif


