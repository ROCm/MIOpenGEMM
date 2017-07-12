/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_PREPGENERATOR_HPP
#define GUARD_MIOPENGEMM_PREPGENERATOR_HPP

#include <miopengemm/basegenerator.hpp>

namespace MIOpenGEMM
{
namespace prepgen
{

class PrepGenerator : public basegen::BaseGenerator
{

  protected:
  
  // TODO why are these not in base generator?
  size_t n_work_items;
  size_t n_work_groups;
  
  Mat::E emat_x;
  char MCHAR;
  char mchar;

  virtual void set_usage() override final;
  void append_basic_what_definitions(std::stringstream& ss);

  virtual size_t get_local_work_size() = 0;
  virtual size_t get_n_work_groups()  = 0;

  size_t get_global_work_size()
  {
    size_t forall_global_work_size = get_n_work_groups() * get_local_work_size();
    return forall_global_work_size;
  }


  public:
  PrepGenerator (Mat::E emat_x,
                const HyperParams&     hp_,
                const Geometry&                     gg_,
                const DerivedParams& dp_);
};
}
}
#endif
