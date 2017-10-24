/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <sstream>
#include <sstream>
#include <miopengemm/basegenerator.hpp>

namespace MIOpenGEMM
{

namespace basegen
{

BaseGenerator::BaseGenerator(const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_)

  : hp(hp_), gg(gg_), dp(dp_), n_args_added(0)
{
}

void BaseGenerator::append_farg(bool u_x, std::stringstream& ss, const std::string& argfrag)
{
  char token = n_args_added == 0 ? ' ' : ',';
  if (u_x == true)
  {
    ss << token << argfrag;
    ++n_args_added;
  }
}

void BaseGenerator::append_fargs(std::stringstream& ss)
{
  ss << "\n(";
  append_farg(u_a, ss, "\n__global const TFLOAT * restrict a, \nconst ulong a_offset");
  append_farg(u_b, ss, "\n__global const TFLOAT * restrict b, \nconst ulong b_offset");
  append_farg(u_c, ss, "\n__global TFLOAT       *          c, \nconst ulong c_offset");
  // if using c, we assume workspace is const.
  // this is a hacky, as we might have a kernel
  // which uses c and modifies w as well.
  std::string cness = (u_c == true) ? "const " : "";
  append_farg(u_w, ss, "\n__global " + cness + "TFLOAT * restrict w,\nconst ulong w_offset");
  append_farg(u_alpha, ss, "\nconst TFLOAT alpha");
  append_farg(u_beta, ss, "\nconst TFLOAT beta");
  ss << ")\n";
}

void BaseGenerator::append_stride_definitions(Mat::E             emat_x,
                                              std::stringstream& ss,
                                              size_t             workspace_type,
                                              bool               withcomments,
                                              std::string        macro_prefix,
                                              bool               with_x_in_name)
{

  char x = Mat::M().name[emat_x];
  if (withcomments == true)
    ss << "/* strides parallel to k (unroll) in " << x << ". MACRO_STRIDE_" << x
       << " is between unroll tiles, STRIDE_" << x << " is within unroll tiles  */\n";

  std::string x_bit = with_x_in_name ? "_" + std::string(1, x) : "";
  for (std::string orth : {"PLL", "PERP"})
  {
    bool pll_k = ("PLL" == orth);
    ss << "#define " << macro_prefix << "STRIDE_" << orth << "_K" << x_bit << " "
       << dp.get_stride(emat_x, pll_k, false, workspace_type) << '\n';
    ss << "#define " << macro_prefix << "MACRO_STRIDE_" << orth << "_K" << x_bit << " "
       << dp.get_stride(emat_x, pll_k, true, workspace_type) << '\n';
  }
}

void BaseGenerator::append_unroll_block_geometry(Mat::E             emat_x,
                                                 std::stringstream& ss,
                                                 bool               withcomments,
                                                 bool               with_x_string)
{

  char        X        = Mat::M().name[emat_x];
  std::string X_string = with_x_string ? "_" + std::string(1, X) : "";

  ss << '\n';
  if (withcomments == true)
  {
    ss << "/* macro tiles define the pattern of C that workgroups "
       << "(threads with shared local memory) process */\n";
  }

  ss << "#define MACRO_TILE_LENGTH" << X_string << " " << dp.at(emat_x).macro_tile_length << '\n';

  if (withcomments == true)
  {
    ss << "/* number of elements in load block : "
       << "MACRO_TILE_LENGTH" << X_string << " * UNROLL */\n";
  }

  ss << "#define N_ELEMENTS_IN" << X_string << "_UNROLL " << dp.at(emat_x).n_elements_in_unroll
     << '\n';

  if (withcomments == true)
  {
    ss << "/* number of groups covering " << (X == 'A' ? 'M' : 'N') << " / MACRO_TILE_LENGTH"
       << X_string;

    if (dp.main_use_edge_trick == 1)
    {
      ss << " + (PRESHIFT_FINAL_TILE" << X_string << " != MACRO_TILE_LENGTH" << X_string << ")";
    }
    ss << " */\n";
  }
  ss << "#define N_GROUPS" << X_string << ' ' << dp.at(emat_x).n_groups << '\n';

  if (dp.main_use_edge_trick != 0)
  {
    if (withcomments == true)
    {
      ss << "/* 1 + (" << (X == 'A' ? 'M' : 'N') << " - 1) % MACRO_TILE_LENGTH" << X_string
         << ". somewhere in 1 ... MACRO_TILE_LENGTH" << X_string << "  */ \n";
    }
    ss << "#define PRESHIFT_FINAL_TILE" << X_string << ' ' << dp.at(emat_x).preshift_final_tile
       << '\n';
  }
}

std::string BaseGenerator::get_time_string()
{

  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  std::time_t generation_time = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << "This " << type << " kernel string was generated on " << std::ctime(&generation_time);

  std::string time_stamp_string = ss.str();

  return "";
  // return stringutil::get_star_wrapped(time_stamp_string.substr(0, time_stamp_string.size() - 1));
}

std::string BaseGenerator::get_what_string()
{
  return stringutil::get_star_wrapped("These parameters define WHAT this kernel does");
}

std::string BaseGenerator::get_how_string()
{
  return stringutil::get_star_wrapped("These parameters define HOW it does it");
}

std::string BaseGenerator::get_derived_string()
{
  return stringutil::get_star_wrapped("The following are implied by preceding: NOT free params!");
}
}
}
