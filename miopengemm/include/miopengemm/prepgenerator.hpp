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
  size_t n_work_items;
  size_t n_work_groups;

  char       matrixchar;
  char       MATRIXCHAR;
  Mat::E emat_x;

  void set_usage_from_matrixchar();
  void append_basic_what_definitions(std::stringstream& ss);

  virtual size_t get_local_work_size() = 0;
  virtual size_t get_n_work_groups()   = 0;

  size_t get_global_work_size()
  {
    size_t forall_global_work_size = get_n_work_groups() * get_local_work_size();
    return forall_global_work_size;
  }

  void initialise_matrixtype(char matrixchar_in)
  {
    if (matrixchar_in == 'a')
    {
      matrixchar = 'a';
      MATRIXCHAR = 'A';
      emat_x     = Mat::E::A;
    }

    else if (matrixchar_in == 'b')
    {
      matrixchar = 'b';
      MATRIXCHAR = 'B';
      emat_x     = Mat::E::B;
    }

    else if (matrixchar_in == 'c')
    {
      matrixchar = 'c';
      MATRIXCHAR = 'C';
      emat_x     = Mat::E::C;
    }

    else
    {
      throw miog_error("in PrepGenerator : unrecognised matrixtype " +
                       std::to_string(matrixchar_in));
    }
  }

  public:
  PrepGenerator(const hyperparams::HyperParams&     hp_,
                const Geometry&                     gg_,
                const derivedparams::DerivedParams& dp_,
                std::string                         type_);
};
}
}
#endif
