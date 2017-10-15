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

KernUses PrepGenerator::get_usage()
{

  KernUses kuses;

  kuses.u_alpha = false;
  kuses.u_k     = false;

  kuses.u_vws.resize(dp.required_workspaces.size(), false);
  for (auto workspace_index = 0; workspace_index < dp.required_workspaces.size(); ++workspace_index)
  {
    if (dp.required_workspaces[workspace_index].emat == emat_x)
    {
      if (dp.required_workspaces[workspace_index].scratch != Scratch::UNUSED)
      {
        kuses.u_vws[workspace_index] = true;
      }
    }
  }

  if (emat_x == Mat::E::C)
  {
    kuses.u_a    = false;
    kuses.u_b    = false;
    kuses.u_c    = true;
    kuses.u_beta = true;
  }

  else
  {
    kuses.u_c    = false;
    kuses.u_beta = false;
    kuses.u_a    = emat_x == Mat::E::A ? true : false;
    kuses.u_b    = !kuses.u_a;
  }

  return kuses;
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
