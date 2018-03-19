/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_BASEGENERATOR_HPP
#define GUARD_MIOPENGEMM_BASEGENERATOR_HPP

#include <miopengemm/derivedparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
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
  bool u_a = false;
  bool u_b = false;
  bool u_c = false;
  bool u_w = false;
  bool u_alpha = false;
  bool u_beta = false;

  std::string get_time_string();
  std::string get_what_string();
  std::string get_how_string();
  std::string get_derived_string();

  virtual size_t get_local_work_size() = 0;
  virtual size_t get_n_work_groups()   = 0;

  private:
  virtual void set_type() = 0;
  void         set_kernelname() { kernelname = "miog_" + type; }

  virtual void set_usage()   = 0;
  virtual void setup_final() = 0;

  public:
  virtual ~BaseGenerator() = default;

  /* Does entire setup. Always called just after construction. */
  void setup()
  {

    set_type();

    set_kernelname();
    set_usage();

    // do anything else which needs to be done.
    setup_final();
  }

  virtual KernBlob get_kernelstring() = 0;

  virtual KType::E get_ktype() = 0;

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
