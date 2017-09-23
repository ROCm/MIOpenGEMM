/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
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
  virtual ~BetacGenerator() = default;
  BetacGenerator(const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_);

  virtual void setup_additional() override final;

  virtual void set_type() override final;

  virtual void append_derived_definitions_additional(std::stringstream& ss) override final;

  size_t get_local_work_size() override final;

  size_t get_work_per_thread() override final;

  virtual KType::E get_ktype() override final;
};

KernBlob get_betac_kernelstring(const HyPas& hp, const Geometry& gg, const DerivedParams& dp);
}
}

#endif
