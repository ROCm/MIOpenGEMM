#ifndef COPYGENERATOR_HPP
#define COPYGENERATOR_HPP

#include <tinygemm/bylinegenerator.hpp>

namespace tinygemm{
namespace copygen{


class CopyGenerator : public bylinegen::ByLineGenerator {

public:

  CopyGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, const std::string & type_);

  void setup_additional() override final;

};


KernelString get_copya_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_copyb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

  
}
}

#endif



