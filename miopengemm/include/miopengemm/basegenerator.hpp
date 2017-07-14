/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_BASEGENERATOR_HPP
#define GUARD_MIOPENGEMM_BASEGENERATOR_HPP

#include <miopengemm/derivedparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/graph.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{
namespace basegen
{
class BaseGenerator
{

  protected:
  const HyPas&         hp;
  const Geometry&      gg;
  const DerivedParams& dp;

  size_t n_args_added;

  // set in virtual function set_type.
  std::string type;

  // set in function set_kernelname.
  std::string kernelname;

  // set in virtual function set_usage.
  bool uses_a;
  bool uses_b;
  bool uses_c;
  bool uses_workspace;
  bool uses_alpha;
  bool uses_beta;

  std::string get_time_string();
  std::string get_what_string();
  std::string get_how_string();
  std::string get_derived_string();

  private:
  virtual void set_type() = 0;
  void         set_kernelname() { kernelname = "miog_" + type; }

  virtual void set_usage()   = 0;
  virtual void setup_final() = 0;

  public:
  /* Does entire setup. Always called just after construction. */
  void setup()
  {

    set_type();
    set_kernelname();
    set_usage();

    // do anything else which needs to be done.
    setup_final();
  }

  // TODO : set type and kernelname. (type_), kernelname("tg_" + type_)
  virtual KernelString get_kernelstring() = 0;

  BaseGenerator(const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_);

  // append argument(s) to the function definition
  void append_farg(bool, std::stringstream&, const std::string&);

  void append_fargs(std::stringstream& ss);

  void append_unroll_block_geometry(Mat::E             emat_x,
                                    std::stringstream& ss,
                                    bool               withcomments,
                                    bool               with_x_string);

  void append_stride_definitions(Mat::E             emat_x,
                                 std::stringstream& ss,
                                 size_t             workspace_type,
                                 bool               withcomments,
                                 std::string        macro_prefix,
                                 bool               append_stride_definitions);
};
}
}

#endif
