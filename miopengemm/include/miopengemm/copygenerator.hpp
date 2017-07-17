/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_COPYGENERATOR_HPP
#define GUARD_MIOPENGEMM_COPYGENERATOR_HPP

#include <miopengemm/bylinegenerator.hpp>

namespace MIOpenGEMM
{
namespace copygen
{

class CopyGenerator : public bylinegen::ByLineGenerator
{

  public:
  CopyGenerator(Mat::E emat_x, const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_);

  virtual void setup_additional() override final;

  virtual void set_type() override final;

  virtual KType::E get_ktype() override final;

  virtual void append_derived_definitions_additional(std::stringstream& ss) override final;

  size_t get_local_work_size() override final;

  size_t get_work_per_thread() override final;
};

KernBlob
get_copy_kernelstring(Mat::E emat_x, const HyPas& hp, const Geometry& gg, const DerivedParams& dp);
}
}

#endif
