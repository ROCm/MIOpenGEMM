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

  bool u_a = false;
  bool u_b = false;
  bool u_c = false;
  bool u_alpha = false;
  bool u_beta = false; 
  bool u_k     = false;
  
  std::vector<bool> u_vws(dp.required_workspaces.size(), false);
  for (auto workspace_index = 0; workspace_index < dp.required_workspaces.size(); ++workspace_index)
  {
    if (dp.required_workspaces[workspace_index].emat == emat_x)
    {
      if (dp.required_workspaces[workspace_index].scratch != Scratch::UNUSED)
      {
        u_vws[workspace_index] = true;
      }
    }
  }

  if (emat_x == Mat::E::C)
  {
    u_a    = false;
    u_b    = false;
    u_c    = true;
    u_beta = true;
  }

  else
  {
    u_c    = false;
    u_beta = false;
    u_a    = emat_x == Mat::E::A ? true : false;
    u_b    = !u_a;
  }

  return KernUses(u_a, u_b, u_c, u_vws, u_alpha, u_beta, u_k) ;
}

void PrepGenerator::append_basic_what_definitions(std::stringstream& ss)
{
  ss << "#define TFLOAT  " << dp.t_float << "\n"
     << "#define LD" << MCHAR << " " << gg.ldX.at(emat_x) << "\n"
  
     << "// less than or equal to LD" << MCHAR << ", DIM_COAL is size in the contiguous direction \n"
     << "// (m for c matrix if col contiguous and not transposed)  \n"
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
