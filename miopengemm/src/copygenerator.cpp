/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <sstream>
#include <miopengemm/copygenerator.hpp>
#include <miopengemm/error.hpp>

namespace MIOpenGEMM
{
namespace copygen
{

CopyGenerator::CopyGenerator(Mat::E               emat_x_,
                             const HyPas&         hp_,
                             const Geometry&      gg_,
                             const DerivedParams& dp_)
  : bylinegen::ByLineGenerator(emat_x_, hp_, gg_, dp_)
{
}

void   CopyGenerator::set_type() { type = "copy" + std::string(1, mchar); }
size_t CopyGenerator::get_local_work_size() { return dp.at(emat_x).cw1_local_work_size; }
size_t CopyGenerator::get_work_per_thread() { return dp.at(emat_x).cw1_work_per_thread; }

KType::E CopyGenerator::get_ktype()
{
  switch (emat_x)
  {
  case Mat::E::A: return KType::E::WSA;
  case Mat::E::B: return KType::E::WSB;
  case Mat::E::C: throw miog_error("C not valid in get_ktype of CopyGenerator");
  case Mat::E::N: throw miog_error("E::N not valid in get_ktype of CopyGenerator");
  }
  throw miog_error("failed in get_ktype");
}

void CopyGenerator::setup_additional()
{
  description_string = R"()";
  inner_work_string  = std::string("\n/* the copy */\nw[i] = ") + std::string(1, mchar) + "[i];";
}

void CopyGenerator::append_derived_definitions_additional(std::stringstream& ss)
{
  if (emat_x != Mat::E::A && emat_x != Mat::E::B)
  {
    std::stringstream errm;
    errm << "Call to append_derived_definitions_additional, "
         << " but mchar is neither a nor b, but it's rather  " << mchar;
    throw miog_error(errm.str());
  }

  ss << "#define LDW " << dp.get_target_ld(emat_x) << "\n";
  ss << "#define GLOBAL_OFFSET_W " << dp.at(emat_x).cw_global_offset << "\n";
}

KernBlob
get_copy_kernelstring(Mat::E emat_x, const HyPas& hp, const Geometry& gg, const DerivedParams& dp)
{

  if (emat_x != Mat::E::A and emat_x != Mat::E::B)
  {
    throw miog_error("get_copy_kernelstring only for A and B matrices");
  }

  CopyGenerator cg(emat_x, hp, gg, dp);
  cg.setup();
  return cg.get_kernelstring();
}
}
}
