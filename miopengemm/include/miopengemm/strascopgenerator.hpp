/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_STRASCOPGENERATOR_HPP
#define GUARD_MIOPENGEMM_STRASCOPGENERATOR_HPP

#include <miopengemm/prepgenerator.hpp>

namespace MIOpenGEMM
{
namespace strasgen
{

class StrasCopGenerator : public prepgen::PrepGenerator
{

  private:
  size_t n_work_items_per_line;

  public:
  StrasCopGenerator(Mat::E emat_x, const HyPas&, const Geometry&, const DerivedParams&, KType::E);
  virtual ~StrasCopGenerator() = default;

  virtual KernBlob get_kernelstring() final override;
  virtual void     setup_final() final override;

  virtual void set_type() override final;
  virtual KType::E get_ktype() override final;

  private:

  protected:
  virtual size_t get_local_work_size() override final;
  virtual size_t get_n_work_groups() override final;

};


KernBlob get_stras_kernelstring(const HyPas&, const Geometry&, const DerivedParams&, KType::E);

}
}

#endif
