/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/prepgenerator.hpp>

namespace MIOpenGEMM
{
namespace prepgen
{

void PrepGenerator::set_usage()
{

  u_alpha = false;
  if (emat_x == Mat::E::C)
  {
    u_a    = false;
    u_b    = false;
    u_c    = true;
    u_w    = false;
    u_beta = true;
  }

  else
  {
    u_c    = false;
    u_w    = true;
    u_beta = false;

    if (emat_x == Mat::E::A)
    {
      u_a = true;
      u_b = false;
    }
    else if (emat_x == Mat::E::B)
    {
      u_a = false;
      u_b = true;
    }

    else
    {
      throw miog_error("Unrecognised emat_x in forallgenerator.cpp");
    }
  }
}

void PrepGenerator::append_basic_what_definitions(std::stringstream& ss)
{
  ss << "#define TFLOAT  " << dp.t_float << "\n"
     << "#define LD" << MCHAR << " " << gg.ldX.at(emat_x) << "\n"
     << "/* less than or equal to LD" << MCHAR
     << ", DIM_COAL is size in the contiguous direction (m for c matrix if col "
     << "contiguous and not transposed) */ \n"
     << "#define DIM_COAL " << gg.get_coal(emat_x) << "\n"
     << "/* DIM_UNCOAL is the other dimension of the matrix */ \n"
     << "#define DIM_UNCOAL " << gg.get_uncoal(emat_x) << "\n\n";
}

PrepGenerator::PrepGenerator(Mat::E               emat_x_,
                             const HyPas&         hp_,
                             const Geometry&      gg_,
                             const DerivedParams& dp_)

  : basegen::BaseGenerator(hp_, gg_, dp_)
{
  emat_x = emat_x_;
  MCHAR  = Mat::M().name[emat_x];
  mchar  = Mat::M().lcase_name[emat_x];
}
}
}
