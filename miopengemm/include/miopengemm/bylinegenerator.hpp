/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_FORALLGENERATOR_HPP
#define GUARD_MIOPENGEMM_FORALLGENERATOR_HPP

#include <miopengemm/prepgenerator.hpp>

namespace MIOpenGEMM
{
namespace bylinegen
{

class ByLineGenerator : public prepgen::PrepGenerator
{

  private:
  size_t n_full_work_items_per_line = 0;
  size_t n_work_items_per_line = 0;
  size_t n_full_work_items = 0;
  size_t start_in_coal_last_work_item = 0;
  size_t work_for_last_item_in_coal = 0;

  protected:
  std::string description_string;
  std::string inner_work_string;

  virtual size_t get_work_per_thread() = 0;

  size_t get_n_work_groups() override final;

  public:
  ByLineGenerator(Mat::E emat_x, const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_);
  virtual ~ByLineGenerator() = default;

  virtual KernBlob get_kernelstring() final override;
  virtual void     setup_final() final override;

  private:
  void append_description_string(std::stringstream& ss);
  void append_how_definitions(std::stringstream& ss);
  void append_copy_preprocessor(std::stringstream& ss);
  void append_derived_definitions(std::stringstream& ss);

  void append_setup_coordinates(std::stringstream& ss);
  void append_positioning_x_string(std::stringstream& ss);
  void append_inner_work(std::stringstream& ss);
  void append_work_string(std::stringstream& ss);
  void append_positioning_w_string(std::stringstream& ss);

  protected:
  virtual void setup_additional()                                           = 0;
  virtual void append_derived_definitions_additional(std::stringstream& ss) = 0;
};
}
}

#endif
