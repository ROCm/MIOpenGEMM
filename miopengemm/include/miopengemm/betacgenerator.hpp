#ifndef GUARD_MIOPENGEMM_BETACGENERATOR_HPP
#define GUARD_MIOPENGEMM_BETACGENERATOR_HPP

#include <sstream>
#include <miopengemm/bylinegenerator.hpp>

namespace MIOpenGEMM
{
namespace betacgen
{

class BetacGenerator : public bylinegen::ByLineGenerator
{

  public:
  BetacGenerator(const hyperparams::HyperParams&     hp_,
                 const Geometry&                     gg_,
                 const derivedparams::DerivedParams& dp_);

  virtual void setup_additional() override final;

  virtual void append_derived_definitions_additional(std::stringstream& ss) override final;

  size_t get_local_work_size() override final;

  size_t get_work_per_thread() override final;
};

KernelString get_betac_kernelstring(const hyperparams::HyperParams&     hp,
                                    const Geometry&                     gg,
                                    const derivedparams::DerivedParams& dp);
}
}

#endif
