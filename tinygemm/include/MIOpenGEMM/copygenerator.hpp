#ifndef COPYGENERATOR_HPP
#define COPYGENERATOR_HPP

#include <MIOpenGEMM/bylinegenerator.hpp>

namespace MIOpenGEMM{
namespace copygen{


class CopyGenerator : public bylinegen::ByLineGenerator {

public:

  CopyGenerator(const hyperparams::HyperParams & hp_,  const Geometry & gg_, const derivedparams::DerivedParams & dp_, const std::string & type_);

  virtual void setup_additional() override final;

  virtual void append_derived_definitions_additional(std::stringstream & ss) override final;

  size_t get_local_work_size() override final;

  size_t get_work_per_thread() override final;
  
};


KernelString get_copya_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp);

KernelString get_copyb_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp);

  
}
}

#endif



