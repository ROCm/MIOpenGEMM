/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <miopengemm/alphagenerator.hpp>
#include <miopengemm/basegenerator.hpp>

/* TODO : reconsider half threads loading A to LDS, others B to LDS, even though
 * will only work in unusual cirmumstances (strides same, or tiles parallel
 * to contiguous). With normal form matrices, it should work in all situations  */
/* TODO : remove final barrier, not nec */
/* TODO : beta = 1 optimisation */
/* TODO : mad as hyper-parameter, figure out why it slows kernels down (see
 * tensile branch) */
/* TODO : float4 */
/* TODO : volatile int id (Nugteren) to prevent unrolling */
/* TODO : play with restrict keyword */
/* TODO : add ICE > 1 with non-interwoven partitioning */

namespace MIOpenGEMM
{
namespace alphagen
{

class AlphaGenerator : public basegen::BaseGenerator
{

  private:
  virtual void set_usage() override final
  {

    /* TODO enum the WOS */ 
    uses_a = (hp.at(Mat::E::A).vs[Chi::E::WOS] == Scratch::E::UNUSED) ? true : false; 
    uses_b = (hp.at(Mat::E::B).vs[Chi::E::WOS] == Scratch::E::UNUSED) ? true : false;
    uses_c = true;
    uses_workspace = (not uses_a or not uses_b);
    uses_alpha = true;
    uses_beta  = dp.main_does_beta_c_inc;
  }

  public:
  AlphaGenerator(const HyperParams&     hp_,
                 const Geometry&                     gg_,
                 const DerivedParams& dp_)
    : basegen::BaseGenerator(hp_, gg_, dp_)
  {
  }

  private:
  void append_group_allocation_string(std::stringstream& ss)
  {
    if (hp.at(Mat::E::C).vs[NonChi::E::GAL] == GroupAllocation::E::BYCOL)
    {
      ss <<
        R"(
/* GROUP_ALLOCATION = 1 :  allocation is done column-by-column */
const TINTA group_id_a = group_id_xy % N_GROUPS_A;
const TINTB group_id_b = group_id_xy / N_GROUPS_A;
)";
    }

    else if (hp.at(Mat::E::C).vs[NonChi::E::GAL] == GroupAllocation::E::BYROW)
    {
      ss <<
        R"(
/* GROUP_ALLOCATION = 2 :  allocation is done row-by-row */
TINTB group_id_b = group_id_xy % N_GROUPS_B;
TINTA group_id_a = group_id_xy / N_GROUPS_B;
)";
    }

    else if (hp.at(Mat::E::C).vs[NonChi::E::GAL] == GroupAllocation::E::SUCOL)
    {
      ss <<
        R"(
/* GROUP_ALLOCATION = 3 : allocation examples
* (if SUPER_COLUMN_WIDTH is 8, m = 3, and N_WORK_ITEMS_PER_C_ELM is 1) is done as follows
* |0   1  2  3  4  5  6  7| 24 25 26
* |8   9 10 11 12 13 14 15| 27 28 29
* |16 17 18 19 20 21 21 23| 30 31 32
*              
* if SUPER_COLUMN_WIDTH is 2 and N_WORK_ITEMS_PER_ELM is 3 it is done as follows
* | (0,   1,  2)  (3,  4,  5 )    |    
* | (6,   7,  8)  (9,  10, 11)    |    ...
* | (12, 13, 14)  (15, 16, 17)    |
*                .
*                .
* where the integers are work group numbers
* */  
TINTB group_id_b;
TINTA group_id_a;
TINTC wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*N_GROUPS_A);
)";

      std::string full_SUCOL_string = R"(
group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
group_id_a = (group_id_xy / SUPER_COLUMN_WIDTH) % N_GROUPS_A;
)";

      std::string partial_SUCOL_string = R"(
group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % LAST_SUPER_COLUMN_WIDTH;
group_id_a = (group_id_xy  - (N_GROUPS_B - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A) / LAST_SUPER_COLUMN_WIDTH;
)";

      // super column width perfectly fits across B
      if (dp.ga3_last_super_column_width == 0)
      {
        ss << full_SUCOL_string;
      }

      else
      {

        // there is just one column
        if (dp.ga3_last_super_column_width == dp.at(Mat::E::B).n_groups)
        {
          ss << partial_SUCOL_string;
        }

        else
        {
          ss << '\n'
             << "if (group_id_xy < (N_GROUPS_B - "
                "LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A){";
          ss << full_SUCOL_string << "}\n";

          ss << "else{";
          ss << partial_SUCOL_string << "}\n";
        }
      }
    }

    else
    {
      std::stringstream err_ss;
      err_ss << "Invalid group_allocation parameter : " << hp.at(Mat::E::C).vs[NonChi::E::GAL]
             << ". It should be one of 1/2/3.";
      throw miog_error(err_ss.str());
    }
  }

  void append_super_column_width_defn(std::stringstream& ss)
  {

    if (hp.at(Mat::E::C).vs[NonChi::E::GAL] == 3)
    {

      ss << "\n\n"
         << "/* This variable defines the width of super-columns (we have "
            "GROUP_ALLOCATION 3). It "
            "is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) "
            "*/\n"
         << "#define SUPER_COLUMN_WIDTH " << dp.ga3_super_column_width;
      ss << "\n/* LAST_SUPER_COLUMN_WIDTH : N_GROUPS_B % SUPER_COLUMN_WIDTH  "
            "*/";
      ss << "\n#define LAST_SUPER_COLUMN_WIDTH " << dp.ga3_last_super_column_width;
    }
  }

  void append_split_on_k_vardecl_write_string(std::stringstream& ss)
  {
    if (dp.main_split_on_k != 0)
    {
      ss <<
        R"(
/* the following variables are used in implementing a basic atomic increment */
global TFLOAT * ptr_to_c_elm;  // with `restrict' is no faster
TFLOAT previous_value; )"
         << '\n'
         << dp.infa << " newVal;\n"
         << dp.infa << " prevVal;"
         << "\n\n";
    }
  }

  void append_loop_var_bound_incr(std::stringstream& ss,
                                  std::string        varname,
                                  std::string        bound_string,
                                  std::string        increment_string, 
                                  Mat::E emat_x)
  {
    ss << "for (TINT" << Mat::M.name[emat_x] << ' ' << varname << " = 0; " << varname << " < " << bound_string << "; "
       << increment_string << ")";
  }

  void append_load_for_perp(Mat::E emat_x, std::stringstream& ss)
  {


    char X = Mat::M.name[emat_x];
    
    std::string bound_string = hp.at(emat_x).vs[Chi::E::LIW] == 0
                                 ? std::string("MICRO_") + X + "_TILE_PERP_UNROLL"
                                 : std::string("MACRO_TILE_LENGTH_") + X;
    std::string increment_string =
      hp.at(emat_x).vs[Chi::E::LIW] == 0
        ? "++mu_perp_i"
        : std::string("mu_perp_i += MACRO_TILE_LENGTH_") + X + "/MICRO_" + X + "_TILE_PERP_UNROLL";
    append_loop_var_bound_incr(ss, "mu_perp_i", bound_string, increment_string, emat_x);
  }

  void append_load_for_pll(Mat::E emat_x, std::stringstream& ss)
  {


    std::string bound_string =
      hp.at(emat_x).vs[Chi::E::LIW] == 0 ? std::string("MICRO_") + Mat::M.name[emat_x] + "_TILE_PLL_UNROLL" : "UNROLL";
    std::string increment_string =
      hp.at(emat_x).vs[Chi::E::LIW] == 0 ? "++mu_pll_i" : std::string("mu_pll_i += UNROLL/MICRO_") +
                                                          Mat::M.name[emat_x] + "_TILE_PLL_UNROLL";
    append_loop_var_bound_incr(ss, "mu_pll_i", bound_string, increment_string, emat_x);
  }

  void append_final_write_element(std::stringstream& ss,
                                  size_t           atomic_increment,
                                  size_t           with_beta_scaling,
                                  size_t           with_alpha_increment)
  {

    // a good place to break kernel to check error checking.
    // make this* 1.11101242345 for example
    std::string alpha_scaled = "alpha*rC[row/C_INTERWEAVE_STRIDE_A][col/C_INTERWEAVE_STRIDE_B]";

    ss << "\nindex = STRIDE_PLL_M_C*(write_start_a + row) + "
          "STRIDE_PLL_N_C*(write_start_b + col);\n";
    // beta string
    ss << (with_beta_scaling == 0 ? "" : "c[index] *= beta;\n");
    if (with_alpha_increment != 0)
    {
      ss << '\n';
      if (atomic_increment == 0)
      {
        ss << "c[index] += " << alpha_scaled << +";\n";
      }

      else
      {
        ss << "ptr_to_c_elm = c + index;\n"
           << "do {\n"
           << "previous_value = *ptr_to_c_elm;\n"
           << "prevVal = as_" << dp.infa << "(previous_value);\n"
           << "newVal = as_" << dp.infa << "(" << alpha_scaled << " + previous_value);\n"
           << "} while (" << dp.fati << "(( __global " << dp.infa
           << "*)(ptr_to_c_elm), prevVal, newVal) != prevVal);";
      }
    }
  }

  void append_for_loops_for_c_write_open(std::stringstream& ss)
  {

    ss << "\n/* loops for writing to c */\n";

    ss << dp.pragma_unroll_string;
    append_loop_var_bound_incr(
      ss,
      "row",
      hp.at(Mat::E::A).vs[Chi::E::MIW] == 0 ? "MICRO_TILE_LENGTH_A" : "MACRO_TILE_LENGTH_A",
      hp.at(Mat::E::A).vs[Chi::E::MIW] == 0 ? "++row" : "row += N_MICRO_IN_MACRO_A", Mat::E::A);
    ss << " {\n";

    ss << dp.pragma_unroll_string;
    append_loop_var_bound_incr(
      ss,
      "col",
      hp.at(Mat::E::B).vs[Chi::E::MIW] == 0 ? "MICRO_TILE_LENGTH_B" : "MACRO_TILE_LENGTH_B",
      hp.at(Mat::E::B).vs[Chi::E::MIW] == 0 ? "++col" : "col += N_MICRO_IN_MACRO_B", Mat::E::B);
    ss << " {\n";
  }

  void append_for_loops_for_c_write_close(std::stringstream& ss) { ss << "\n}\n}\n"; }

  void append_check_wrapped_if_clause_open(std::stringstream& ss)
  {
    ss <<
      R"(
/* catching the write cases */
if (
/* B overflow, but not A edge */
((write_start_b + col >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1)) && group_id_a  !=  (N_GROUPS_A - 1 )) ||
/* A overflow, but not B edge */
((write_start_a + row >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)) && group_id_b  !=  (N_GROUPS_B - 1 )) ||
/* A edge and B edge and A overflow and B overflow */
(
group_id_a == (N_GROUPS_A - 1)   && 
group_id_b == (N_GROUPS_B - 1)   && 
write_start_b + col >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1) && 
write_start_a + row >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)
)){
)";
  }

  void append_check_wrapped_if_clause_close(std::stringstream& ss) { ss << "\n}"; }

  void append_checked_wrapped_loops_from_bools(std::stringstream& ss,
                                               size_t           with_check,
                                               size_t           atomic_increment,
                                               size_t           with_beta_scaling,
                                               size_t           with_alpha_increment)
  {

    append_for_loops_for_c_write_open(ss);
    if (with_check != 0)
    {
      append_check_wrapped_if_clause_open(ss);
      append_final_write_element(ss, atomic_increment, with_beta_scaling, with_alpha_increment);
      append_check_wrapped_if_clause_close(ss);
    }

    else
    {
      append_final_write_element(ss, atomic_increment, with_beta_scaling, with_alpha_increment);
    }
    append_for_loops_for_c_write_close(ss);
  }

  void append_final_write_loops(std::stringstream& ss, size_t with_check)
  {
    if (dp.main_split_on_k == 0)
    {
      append_checked_wrapped_loops_from_bools(ss, with_check, 0, 1, 1);
    }

    else
    {
      append_checked_wrapped_loops_from_bools(ss, with_check, 1, 0, 1);
    }
  }

  void append_final_write_loops_no_check(std::stringstream& ss) { append_final_write_loops(ss, 0); }

  void append_final_write_loops_with_check(std::stringstream& ss)
  {
    append_final_write_loops(ss, 1);
  }

  void append_k_remaining_string(std::stringstream& ss)
  {
    ss << '\n' << "TSHORT k_remaining = " << dp.effective_k_varies_string << " % UNROLL;";
  }

  // simple for loops. Could consider unrolling like Cobalt, but for the moment
  // I use the optional pragma unroll
  void append_load_ab_into_LDS_string(std::stringstream& ss,
                                      size_t           final_unroll,
                                      size_t           special_first_unroll)
  {

    append_load_into_LDS_string(Mat::E::A, ss, final_unroll, special_first_unroll);
    append_load_into_LDS_string(Mat::E::B, ss, final_unroll, special_first_unroll);

    ss <<
      R"(
/* make sure all loads from LDS memory have completed */
barrier(CLK_LOCAL_MEM_FENCE); )";
  }

  // simple for loops. Could consider unrolling like Cobalt,
  // but for the moment I use the optional pragma unroll
  void append_load_into_LDS_string(Mat::E emat_x,
                                   std::stringstream& ss,
                                   size_t           final_unroll,
                                   size_t           special_first_unroll)
  {

    char X = Mat::M.name[emat_x];
    char x = Mat::M.lcase_name[emat_x];

    std::string n_jumps_string = dp.main_split_on_k == 0 ? "UNROLL" : "G_UNROLL";

    if (final_unroll != 0 && special_first_unroll != 0)
    {
      throw miog_error("From get_load_ab_into_LDS_string > It is not possible "
                       "for this to be both "
                       "a `special_first_unroll' and a `final_unroll'. This is "
                       "a logic error, "
                       "broken alg, come and sort it out");
    }

    std::stringstream ss_value_to_get;
    std::stringstream ss_comment;

    if (final_unroll == 1 || special_first_unroll == 1)
    {
      std::string condition       = final_unroll == 1 ? " < k_remaining " : " >= unroll_offset";
      std::string special_comment = final_unroll == 1 ? "(ignoring tail)" : "(ignoring prepend)";
      ss_value_to_get << "(" << x << "_offset_pll_unroll + mu_pll_i) " << condition << " ? " << x
                      << "[mu_pll_i*STRIDE_PLL_K_" << X << " + mu_perp_i*STRIDE_PERP_K_" << X
                      << "] : 0;";
      ss_comment << "/* load final bit of data from " << x << " into LDS, less than a full unroll "
                 << special_comment << " */";
    }

    else
    {
      ss_value_to_get << x << "[mu_pll_i*"
                      << "STRIDE_PLL_K_" << X << " + mu_perp_i*"
                      << "STRIDE_PERP_K_" << X << "];";
      ss_comment << "/* load data from " << x << " into LDS */";
    }

    ss << '\n' << ss_comment.str() << '\n' << dp.pragma_unroll_string;
    append_load_for_perp(emat_x, ss);
    ss << " {\n" << dp.pragma_unroll_string;
    append_load_for_pll(emat_x, ss);
    ss << " {\n"
       << "local" << X << "[MACRO_TILE_LENGTH_" << X << "_AND_PAD*(" << x
       << "_offset_pll_unroll + mu_pll_i) + (" << x << "_offset_perp_unroll + mu_perp_i)] = \n"
       << ss_value_to_get.str() << '\n'
       << "}\n"
       << "}\n";

    if (final_unroll == 0)
      ss << x << " += "
         << "STRIDE_PLL_K_" << X << "*" << n_jumps_string << ";\n";

    ss << '\n';
  }

  std::string get_c_work_item_next(Mat::E emat_x)
  {

    return (hp.at(emat_x).vs[Chi::E::MIW] != 0) ? "1" : (std::string("MICRO_TILE_LENGTH_") + Mat::M.name[emat_x]);
  }

  // We previously had a variable unroll_the_math_section = False.
  // Experiments with unroll_the_math_section suggest that it's a bad idea.
  void append_math_section(std::stringstream& ss, size_t use_k_remaining)
  {

    std::string number_of_unrolls = use_k_remaining == 0 ? "UNROLL" : "k_remaining";
    ss << "\nfor (TSHORT u = 0; u < " << number_of_unrolls << "; ++u){\n";
    append_load_to_register_string(Mat::E::A, ss);
    append_load_to_register_string(Mat::E::B, ss);
    ss << '\n';
    append_compute_string(ss);

    ss << "}\n";
  }

  void append_relocate_load_math_string(std::stringstream& ss,
                                        size_t           final_unroll,
                                        size_t           special_first_unroll)
  {
    if (final_unroll != 0 && special_first_unroll != 0)
    {
      throw miog_error("From get_relocate_load_math_string : It is not "
                       "possible for this to be "
                       "both a `special_first_unroll' and a `final_unroll'. "
                       "This is a logic error, "
                       "broken alg, come and sort it out");
    }

    append_load_ab_into_LDS_string(ss, final_unroll, special_first_unroll);

    ss << '\n';
    for (Mat::E emat_x : {Mat::E::A, Mat::E::B})
    {
      char X = Mat::M.name[emat_x];
      char x = Mat::M.lcase_name[emat_x];

      ss << '\n'
         << "l" << X << " = local" << X << " + micro_id_" << x << "*" << get_c_work_item_next(emat_x)
         << ";";
    }

    ss << '\n';

    append_math_section(ss, final_unroll);
    ss <<
      R"(
/* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
barrier(CLK_LOCAL_MEM_FENCE); )";
  }

  void append_final_unroll_string(std::stringstream& ss)
  {

    if (dp.main_split_on_k == 0)
    {
      ss << '\n';
      append_relocate_load_math_string(ss, 1, 0);
      ss << '\n';
    }
    else
    {
      ss <<
        R"(
/* There is one workgroup which will process the remainder (less that UNROLL) */
/* JN 16 Nov 2016 */
if (group_id_z == n_work_groups_with_1_more && k_remaining > 0){
)";
      append_relocate_load_math_string(ss, 1, 0);
      ss << "\n}\n";
    }
  }

  void append_first_unroll_block(std::stringstream& ss)
  {
    if (hp.at(Mat::E::C).vs[NonChi::E::UFO] != 0)
    {
      ss << "\n\n/* This is where the first unroll will be performed. "
            "Identical to what is in the "
            "main while, but with zero buffering.  */";
      if (dp.main_split_on_k == 0)
      {
        ss << '\n';
        append_relocate_load_math_string(ss, 0, 1);
        ss << "\n--n_unrolls_remaining;\n";
      }
      else
      {
        ss << "\nif (group_id_z == 0){\n";
        append_relocate_load_math_string(ss, 0, 1);
        ss << "\n--n_unrolls_remaining;\n}";
      }
    }
  }

  void append_compute_string(std::stringstream& ss)
  {
    ss << dp.pragma_unroll_string << "for (TSHORT row = 0; row < MICRO_TILE_LENGTH_A; ++row){\n"
       << dp.pragma_unroll_string << "for (TSHORT col = 0; col < MICRO_TILE_LENGTH_B; ++col){\n"
       << "rC[row][col] += rA[row]*rB[col];   \n}\n}\n";
  }

  void append_load_to_register_string(Mat::E emat_x, std::stringstream& ss)
  {
    char X = Mat::M.name[emat_x];
    
    ss << '\n' << dp.pragma_unroll_string;
    ss << "for (TSHORT i = 0; i < MICRO_TILE_LENGTH_" << X << "; ++i){\n";
    ss << "r" << X << "[i] = l" << X << "["
       << "i*"
       << "C_INTERWEAVE_STRIDE_" << X << "];\n}\n";
    ss << "l" << X << " += MACRO_TILE_LENGTH_" << X << "_AND_PAD;\n";
  }

  void append_group_allocation_defn_string(std::stringstream& ss)
  {
    ss << "#define GROUP_ALLOCATION " << hp.at(Mat::E::C).vs[NonChi::E::GAL] << '\n';
    if (hp.at(Mat::E::C).vs[NonChi::E::GAL] == 3)
    {
      ss << "/* this variable is declared because we have GROUP_ALLOCATION "
            "type 3. */\n";
      ss << "/* It should define how many workgroups we expect to have active "
            "simulantaneuosly. "
            "*/\n";
      ss << "#define N_TARGET_ACTIVE_WORKGROUPS " << hp.at(Mat::E::C).vs[NonChi::E::NAW] << '\n';
    }
  }

  void append_final_write_all(std::stringstream& ss)
  {

    if (dp.main_use_edge_trick == 0)
    {
      ss << '\n';
      append_final_write_loops_no_check(ss);
    }

    else
    {

      std::string cond_a("");
      if (dp.at(Mat::E::A).preshift_final_tile != dp.at(Mat::E::A).macro_tile_length)
      {
        cond_a = "(group_id_a != N_GROUPS_A - 1)";
      }

      std::string cond_b("");
      if (dp.at(Mat::E::B).preshift_final_tile != dp.at(Mat::E::B).macro_tile_length)
      {
        cond_b = "(group_id_b != N_GROUPS_B - 1)";
      }

      if (cond_a == "" && cond_b == "")
      {
        append_final_write_loops_no_check(ss);
      }

      else
      {

        if (dp.main_use_edge_trick == 0)
        {
          throw miog_error("in alphagenerator, dp.main_use_edge_trick == 0. "
                           "however, non-perfectly tilable");
        }

        ss << "\n/* the case where this is not an edge tile : will write to "
              "all cells */ \n";
        if (cond_a != "" && cond_b != "")
        {
          ss << "if (" << cond_a << " && " << cond_b << "){ \n";
        }
        else if (cond_a != "")
        {
          ss << "if  " << cond_a << "{ \n";
        }
        else
        {
          ss << "if  " << cond_b << "{ \n";
        }
        append_final_write_loops_no_check(ss);
        ss << "\n}";
        ss << "\n\nelse{";
        append_final_write_loops_with_check(ss);
        ss << "\n}";
      }
    }
  }

  void append_split_on_k_defns_string(std::stringstream& ss)
  {
    if (dp.main_split_on_k != 0)
    {
      ss <<
        R"(/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL )"
         << hp.at(Mat::E::C).vs[NonChi::E::ICE] * hp.at(Mat::E::C).vs[NonChi::E::UNR]
         << " // N_WORK_ITEMS_PER_C_ELM*UNROLL";
    }
  }

  void append_group_id_defns(std::stringstream& ss)
  {
    if (dp.main_split_on_k == 0)
    {
      ss << "\nconst TINTC group_id_xy = get_group_id(0);\n";
    }
    else
    {
      ss <<
        R"(
const TINTC group_id = get_group_id(0);
const TINTC group_id_xy = group_id / N_WORK_ITEMS_PER_C_ELM;
const TSHORT group_id_z = group_id % N_WORK_ITEMS_PER_C_ELM;
)";
    }
  }

  void append_stride_c_defn(std::stringstream& ss)
  {

    size_t transposed_xor_is_col_major = (gg.tX[Mat::E::C] + gg.isColMajor) % 2;
    ss << "#define STRIDE_PLL_M_C " << (transposed_xor_is_col_major == 1 ? 1 : gg.ldX[Mat::E::C])
       << '\n';
    ss << "#define STRIDE_PLL_N_C " << (transposed_xor_is_col_major == 0 ? 1 : gg.ldX[Mat::E::C])
       << '\n';
  }

  void append_n_unrolls_remaining_string(std::stringstream& ss)
  {
    std::string k_effective_mod_G_UNROLL = dp.effective_k_varies_string + " % G_UNROLL";
    std::string k_effective_div_G_UNROLL = dp.effective_k_varies_string + " / G_UNROLL";
    std::string k_effective_div_UNROLL   = dp.effective_k_varies_string + " / UNROLL";

    if (dp.main_split_on_k == 0)
    {
      ss << "\nint n_unrolls_remaining = " << k_effective_div_UNROLL << ";";
    }

    else
    {
      ss << "\n/* a certain number of work groups process one more unroll. "
            "Note that with UFO = 1, "
            "this depends on column */";
      ss << "\nconst int n_work_groups_with_1_more = (" << k_effective_mod_G_UNROLL
         << ") / UNROLL; \n";
      ss << "\n/* branching between work groups : some wgs have 1 more unroll "
            "to process. */\n";
      ss << "int n_unrolls_remaining = (" << k_effective_div_G_UNROLL;
      ss << ") +  (group_id_z < n_work_groups_with_1_more);";
    }
  }

  void append_c_offset_string(std::stringstream& ss)
  {

    ss << R"(

/* In OpenCL, host code does not have access to raw data pointers. */
/* Host code works with cl_mem objects, which encapsulate and hide raw points. */
/* For this reason, host code CANNOT simply increment pointers to data, */
/* as one can do with pointers for CPU gemm, or cublas gemm.*/

c += c_offset;
)";
  }

  void append_id_string_nonsym(std::stringstream& ss)
  {
    ss << "const TSHORT local_id = get_local_id(0);\n";
    append_group_id_defns(ss);

    ss << R"(
/* Define which part of the C macro-tile this thread will process (% / or / % ? doesn't seem to make much difference) */
const TSHORT micro_id_a = local_id % N_MICRO_IN_MACRO_A;
const TSHORT micro_id_b = local_id / N_MICRO_IN_MACRO_A;
)";

    append_group_allocation_string(ss);

    if (hp.at(Mat::E::C).vs[NonChi::E::UFO] != 0)
    {
      ss <<
        R"(
/* this additional offset of a and b appears because UNROLL_FOR_OFFSET is 1 */
TSHORT unroll_offset = (13*group_id_a + 7*group_id_b)%UNROLL;
TINTK k_plus_offset = __K__ + unroll_offset;
)";
    }
  }

  void append_id_string_sym(std::stringstream& ss, Mat::E emat_x)
  {

    char X = Mat::M.name[emat_x];
    char x = Mat::M.lcase_name[emat_x];
    
    ss << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* LDS memory */\n";
    ss << "__local TFLOAT local" << X << "[N_ELEMENTS_IN_PADDED_" << X << "_UNROLL];\n";
    if (emat_x == Mat::E::A)
      ss << "/* jumping pointer to locate the LDS to load into register memory "
            "*/\n";
    ss << "__local const TFLOAT * l" << X << ";\n";
    if (emat_x == Mat::E::A)
      ss << "/* register memory */ \n";
    ss << "TFLOAT r" << X << "[MICRO_TILE_LENGTH_" << X << "];\n";
    if (emat_x == Mat::E::A)
      ss << "/* Define which part of the C macro-tile this thread will process "
            "(% / or / % ? "
            "doesn't seem to make much difference) */\n";
    ss << "TINT" << X << " write_macro_tile_start_" << x << " = group_id_" << x << "*MACRO_TILE_LENGTH_"
       << X << "; \n";
    if (dp.main_use_edge_trick != 0)
    {
      if (emat_x == Mat::E::A)
        ss << "/* tile on edge : pulling it in so no C overflow */\n";
      ss << "if (group_id_" << x << " == N_GROUPS_" << X << " - 1){\n";
      ss << "write_macro_tile_start_" << x << " -= (MACRO_TILE_LENGTH_" << X
         << " - PRESHIFT_FINAL_TILE_" << X << ");\n";
      ss << "}\n";
    }
    ss << "const TINT" << X << " write_start_" << x << " = write_macro_tile_start_" << x << " + micro_id_"
       << x << "*" << get_c_work_item_next(emat_x) << ";\n";

    ss << "\n\n\n";

    if (hp.at(emat_x).vs[Chi::E::WOS] == Scratch::E::COPY || hp.at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      if (emat_x == Mat::E::A)
        ss << "/* from workspace */\n";
      ss << "const TFLOAT * restrict " << x << " = w + w_offset + GLOBAL_OFFSET_" << X << ";\n";
    }

    else
    {
      ss << x << " += " << x << "_offset;\n";
    }

    if (emat_x == Mat::E::A)
      ss << "/* Define what of A this thread will load from unroll tile in "
            "global to LDS (% / or / "
            "% ? looks like no difference ) */\n";
    ss << "const TINT" << X << " pll_unroll_" << x << "_load_id = local_id % N_MICRO_" << X
       << "_TILES_PLL_UNROLL;\n";
    ss << "const TINT" << X << " perp_unroll_" << x << "_load_id = local_id / N_MICRO_" << X
       << "_TILES_PLL_UNROLL;\n";

    if (emat_x == Mat::E::A)
      ss << "/* Define which part of A this thread will read from (% / "
            "or / % ? doesn't "
            "seem to make much difference) */\n";
    ss << "TINT" << X << " read_macro_tile_start_" << x << " = group_id_" << x << "*MACRO_TILE_LENGTH_"
       << X << "; \n";
    if (dp.main_use_edge_trick != 0 && hp.at(emat_x).vs[Chi::E::WOS] != Scratch::E::NFORM)
    {
      if (emat_x == Mat::E::A)
        ss << "/* tile on edge and A is not normal form: pulling in read zone "
              "so no C overflow */\n";
      ss << "if (group_id_" << x << " == N_GROUPS_" << X << " - 1){\n";
      ss << "read_macro_tile_start_" << x << " -= (MACRO_TILE_LENGTH_" << X
         << " - PRESHIFT_FINAL_TILE_" << X << ");\n";
      ss << "}\n";
    }

    if (emat_x == Mat::E::A)
      ss << "/* move to corner of the region required by the macro tile */\n";
    ss << x << " += read_macro_tile_start_" << x << "*MACRO_STRIDE_PERP_K_" << X << ";\n";

    if (dp.main_split_on_k != 0)
    {
      if (emat_x == Mat::E::A)
        ss << R"(/* a points to top left of region required, but this work group  */
/* might not process the whole of a. So turn 90 and move to the start for this wg */
)";
      ss << x << " += UNROLL*group_id_z*STRIDE_PLL_K_" << X << ";\n";
    }

    if (hp.at(Mat::E::C).vs[NonChi::E::UFO] != 0)
    {
      if (emat_x == Mat::E::A)
        ss << "/* UFO != 0, so offsetting the unroll */\n";
      ss << x << " -= unroll_offset*STRIDE_PLL_K_" << X << ";\n";
    }

    std::string str_n_pll("");
    std::string str_n_perp("");
    if (hp.at(emat_x).vs[Chi::E::LIW] == 0)
    {
      str_n_pll  = std::string("MICRO_") + X + "_TILE_PLL_UNROLL *";
      str_n_perp = std::string("MICRO_") + X + "_TILE_PERP_UNROLL *";
    }
    if (emat_x == Mat::E::A)
      ss << "/* make the micro adjustments (A) for the thread, getting ready "
            "to load */\n";
    ss << "const TINT" << X << " " << x << "_offset_pll_unroll = " << str_n_pll << " pll_unroll_" << x
       << "_load_id;\n";
    ss << "const TINT" << X << " " << x << "_offset_perp_unroll = " << str_n_perp << " perp_unroll_" << x
       << "_load_id;\n";
    ss << x << " += "
       << "STRIDE_PLL_K_" << X << " * " << x << "_offset_pll_unroll;\n";
    ss << x << " += "
       << "STRIDE_PERP_K_" << X << " * " << x << "_offset_perp_unroll;\n";

    ss << '\n';
  }

  void append_transpose_note(std::stringstream& ss)
  {
    ss << R"(
/* A note on how transposes isColMajor  effect the kernel generated: very little. */
/* just via STRIDE_PLL_K_{A,B}, STRIDE_PERP_K_{A,B}, STRIDE_PLL_M_C, STRIDE_PERP_M_C */
)";
  }

  void add_predefine_chiral(Mat::E emat_x, std::stringstream& ss)
  {

    // TODO : should be X ... (upper case) ...
    char x = Mat::M.name[emat_x];

    auto defcom = [emat_x, &ss](std::string&& comment) {
      if (emat_x == Mat::E::A)
        ss << "/*"
           << " " << comment << " : */\n";
    };

    bool withcomments  = emat_x == Mat::E::A;
    
    bool with_x_in_name = true;
    append_unroll_block_geometry(emat_x, ss, withcomments, with_x_in_name);

    append_stride_definitions(emat_x, ss, hp.at(emat_x).vs[Chi::E::WOS], withcomments, "", with_x_in_name);

    if (emat_x == Mat::E::A)
      ss << "/* micro tiles define the pattern of C that individual threads "
            "process */\n";
    ss << "#define MICRO_TILE_LENGTH_" << x << " " << hp.at(emat_x).vs[Chi::E::MIC] << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* the amount of padding of " << x
         << " in LDS (local) memory, to avoid bank comflicts */\n";
    ss << "#define PAD_LDS_" << x << "  " << hp.at(emat_x).vs[Chi::E::PAD] << '\n';
    if (emat_x == Mat::E::A)
      ss << "/* whether loading of " << x << " from global should try to be long in direction of "
                                             "unroll (1) or perpendicular to it (0) */\n";
    ss << "#define WORK_ITEM_LOAD_" << x << "_PLL_TO_UNROLL " << hp.at(emat_x).vs[Chi::E::PLU]
       << '\n';
    defcom("MACRO_TILE_LENGTH_A + PAD_LDS_A");
    ss << "#define MACRO_TILE_LENGTH_" << x << "_AND_PAD "
       << dp.at(emat_x).main_macro_tile_length_and_pad << '\n';

    defcom("MACRO_TILE_LENGTH_A_AND_PAD * UNROLL");
    ss << "#define N_ELEMENTS_IN_PADDED_" << x << "_UNROLL "
       << dp.at(emat_x).main_n_elements_in_padded_unroll << '\n';

    defcom("N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP");
    ss << "#define N_ELEMENTS_OF_" << x << "_TO_LOAD_PER_WORKITEM "
       << dp.at(emat_x).main_n_elements_to_load_per_workitem << '\n';
    defcom("MACRO_TILE_LENGTH_A / MICRO_TILE_LENGTH_A");
    ss << "#define N_MICRO_IN_MACRO_" << x << "  " << dp.at(emat_x).main_n_micro_in_macro << '\n';
    defcom("MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = "
           "N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM");
    ss << "#define MICRO_" << x << "_TILE_PLL_UNROLL " << dp.at(emat_x).main_micro_tile_pll_unroll
       << " \n";
    ss << "#define MICRO_" << x << "_TILE_PERP_UNROLL " << dp.at(emat_x).main_micro_tile_perp_unroll
       << '\n';

    defcom("MACRO_TILE_LENGTH_A / MICRO_A_TILE_PLL_UNROLL");
    ss << "#define N_MICRO_" << x << "_TILES_PLL_UNROLL "
       << dp.at(emat_x).main_n_micro_tiles_pll_unroll << " \n";
    if (emat_x == Mat::E::A)
      ss << "/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if "
            "the load tiles are "
            "truly contiguous tiles (0) */\n";
    ss << "#define LOAD_TO_LDS_INTERWOVEN_" << x << " " << hp.at(emat_x).vs[Chi::E::LIW] << '\n';
    if (emat_x == Mat::E::A)
      ss << "/* Whether micro tile being processed by a compute item is "
            "interwoven with other "
            "micro tiles (ala Cobalt, (1)) or if the micro tiles are "
            "contiguous in C */\n";
    ss << "#define C_MICRO_TILES_INTERWOVEN_" << x << " " << hp.at(emat_x).vs[Chi::E::MIW] << '\n';

    if (emat_x == Mat::E::A)
      ss << "/* depending on whether loads to c are interwoven, set as MIW == "
            "0 ? 1 : "
            "N_MICRO_IN_MACRO_A */\n";
    ss << "#define C_INTERWEAVE_STRIDE_" << x << " " << dp.at(emat_x).main_c_interweave_stride
       << '\n';


    if (hp.at(emat_x).vs[Chi::E::WOS] != Scratch::E::UNUSED)
    {
      if (emat_x == Mat::E::A)
        ss << "/* global memory offset, depends on type of copy of both a,b "
              "*/\n";
      ss << "#define GLOBAL_OFFSET_" << x << " " << dp.at(emat_x).cw_global_offset;
    }

    ss << '\n';
  }

  public:
  // the "main" kernel
  virtual KernelString get_kernelstring() override final
  {

    std::stringstream ss;
    ss << get_time_string();
    ss << "\n\n";
    ss << "/* this kernel was generated for starting geometry : */\n";
    ss << "/* " << gg.get_string() << "*/\n";
    ss << "#define __K__ " << gg.k << '\n';
    ss << "#define TFLOAT  " << dp.t_float << '\n';
    ss << "#define DOES_BETA_C_INC " << dp.main_does_beta_c_inc << '\n';
    ss << "#define DOES_ALPHA_A_B_INC 1"
       << '\n';

    append_transpose_note(ss);

    ss <<
      R"(

)";

    for (auto emat_x : {Mat::E::A, Mat::E::B})

    {
      ss << "\n/* ********************************** specific to " << Mat::M.name[emat_x]
         << " *************************************** */";
      add_predefine_chiral(emat_x, ss);
    }
    
    ss << "\n/* integer types for navigating each of the memory buffers */\n";
    for (size_t i = 0; i < Mem::E::N; ++i){
      ss << "#define TINT" << Mem::M.name[i] << " " << dp.tints[i] << '\n';
    }
    ss << "\n/* type for integer in inner most loops (probably inlined anyway)  */\n";
    ss << "#define TSHORT " << dp.tshort << '\n';
    ss << "\n/* type for integers which never exceeds __K__ + UNROLL (for UFO case) */\n";
    ss << "#define TINTK " << dp.tintk << '\n';

    ss << "\n/* ********************************** common to A and B "
          "*************************************** */";
    ss << "\n/* whether or not to shimmy the starting k, in an attempt to "
          "avoid cache line overuse "
          "for cases where lda/ldb are powers of 2 */\n";
    ss << "/* if 0, no shimmying. if 1, instead of starting at k = 0 "
          "workgroups start at some "
          "negative offset dependent on work group id */\n";
    ss << "/* in the same way as the final unroll populates LDS with zeros in "
          "k mod UNROLL != 0, "
          "the initial negative indices here populate with 0 */\n";
    ss << "#define UNROLL_FOR_OFFSET " << hp.at(Mat::E::C).vs[NonChi::E::UFO] << '\n';

    ss << "/* How much a workgroup loads (global -> LDS) in the k-direction at "
          "each iteration of "
          "the outer-most loop */\n";
    ss << "#define UNROLL " << hp.at(Mat::E::C).vs[NonChi::E::UNR] << '\n';
    ss << "/* whether or not this kernel uses the edge trick (SC17 submission) "
          "*/\n";
    ss << "/* this precompiler defn has no direct influence on the running the "
          "kernel, "
          "implementation already done in make_kernel.py */\n";
    ss << "#define EDGETRICK " << dp.main_use_edge_trick << '\n';
    ss << "/* the number of work items working on the same c element. if this "
          "is 1, there will be "
          "just one thread doing all k multiply-adds, */\n";
    ss << "/* otherwise if it is greater than 1, each thread will be computing "
          "~ k / "
          "N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added "
          "at the end */ \n";
    ss << "#define N_WORK_ITEMS_PER_C_ELM " << hp.at(Mat::E::C).vs[NonChi::E::ICE] << '\n';

    ss << R"(/* define the way in which work groups are assigned to tiles */
/* 1 : column-by-column
 * 2 : row-by-row 
 * 3 : by rows within super-column  */
)";

    append_group_allocation_defn_string(ss);

    ss << "/* Whether to use the unroll pragma to encourage the compiler to "
          "unroll certain loops "
          "*/\n";
    ss << "/* Included here for user, in practice it has no direct effect on "
          "this kernel, as the "
          "relevent implementation has been done in make_kernel.py */\n";
    ss << "#define PRAGMA_UNROLL_FORLOOPS " << hp.at(Mat::E::C).vs[NonChi::E::PUN] << '\n';
    ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many "
          "steps ahead are we "
          "reading into registers, as compared to doing the math. */\n";
    ss << "/* This should be the domain of the compiler, and currently "
          "(26/08/2016, Catalyst) it "
          "seems that leaving this as 0 is best.  */\n";
    ss << "#define N_PREFETCH_FOR_REGISTER_LOAD " << 0 << '\n';
    ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many "
          "steps ahead are we "
          "reading into LDS, as compared to the unroll loop */\n";
    ss << "/* This should be the domain of the compiler, and currently "
          "(26/08/2016, Catalyst) it "
          "seems that leaving this as 0 is best.  */\n";
    ss << "#define N_PREFETCH_FOR_LDS_LOAD " << 0 << '\n';
    ss << "#define MACRO_TILE_AREA " << dp.main_macro_tile_area << '\n';
    ss << "#define MICRO_TILE_AREA " << dp.main_micro_tile_area << '\n';
    ss << "#define N_WORK_ITEMS_PER_WORKGROUP  " << dp.main_n_work_items_per_workgroup << '\n';
    ss << "/* two more parameters, which do dot have an effect the running of "
          "this kernel (used in "
          "enqueuing) */\n";
    ss << "/* the total number of work groups this kernel will use (recall "
          "m,n,k are fixed) */ \n";
    ss << "/* N_WORK_ITEMS_PER_C_ELM * ((M/MACRO_TILE_LENGTH_A) + "
          "(M%MACRO_TILE_LENGTH_A != 0)) * "
          "((N/MACRO_TILE_LENGTH_B) + (N%MACRO_TILE_LENGTH_B != 0)) */ \n";
    ss << "#define N_WORK_GROUPS " << dp.main_n_work_groups << '\n';
    ss << "/* the global work size, ie the total mumber of work items "
          "(threads) which will run */\n ";
    ss << "/* N_WORK_GROUPS * N_WORK_ITEMS_PER_WORKGROUP */ \n";
    ss << "#define GLOBAL_WORK_SIZE " << dp.main_global_work_size << '\n';

    append_stride_c_defn(ss);
    append_split_on_k_defns_string(ss);
    append_super_column_width_defn(ss);

    ss << "\n\n\n__attribute__((reqd_work_group_size(" << dp.main_n_work_items_per_workgroup
       << ",1, 1)))\n";
    ss << "__kernel void ";
    ss << kernelname;
    append_fargs(ss);

    ss << "\n{\n\n";

    append_c_offset_string(ss);

    append_id_string_nonsym(ss);
    ss << "\n\n/* *************************** A setup "
          "************************** */";
    append_id_string_sym(ss, Mat::E::A);
    ss << "\n\n/* *************************** B setup "
          "************************** */";
    append_id_string_sym(ss, Mat::E::B);

    ss << "\n\n\n";

    ss << "/* register memory for C */\n ";
    ss << "TFLOAT rC[MICRO_TILE_LENGTH_A][MICRO_TILE_LENGTH_B] = {{0.}};\n";

    append_n_unrolls_remaining_string(ss);
    append_first_unroll_block(ss);

    ss << "\n\nwhile (n_unrolls_remaining > 0){\n";
    append_relocate_load_math_string(ss, 0, 0);
    ss << "\n--n_unrolls_remaining;\n}\n";

    if (dp.main_final_fractional_unroll == 1)
    {
      ss << "\n/* *********** processing the tail *************** */\n";
      append_k_remaining_string(ss);
      append_final_unroll_string(ss);
      ss << "\n/* *********************************************** */\n\n";
    }

    ss << "\n\n";
    ss << "TINTC index;\n";
    append_split_on_k_vardecl_write_string(ss);
    append_final_write_all(ss);
    ss << "\n}\n";

    return {{uses_a, uses_b, uses_c, uses_workspace, uses_alpha, uses_beta},
            ss.str(),
            kernelname,
            dp.main_global_work_size,
            dp.main_n_work_items_per_workgroup};
  }


virtual void set_type() override final{
  type = dp.main_does_beta_c_inc ? "betac_alphaab" : "alphaab";
}

virtual void setup_final() override final{
  
}

};



KernelString get_alpha_kernelstring(const HyperParams&     hp,
                                    const Geometry&                     gg,
                                    const DerivedParams& dp)
{
  AlphaGenerator ag(hp, gg, dp);
  ag.setup();
  return ag.get_kernelstring();
}
}
}
