#include <tinygemm/alphagenerator.hpp>
#include <tinygemm/generatorutil.hpp>
#include <tinygemm/basegenerator.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <tuple>
#include <fstream>




namespace tinygemm{
namespace alphagen{


class AlphaGenerator : basegen::BaseGenerator{

private:


  



void set_usage(){
  
  uses_a = (hp.normal_form || hp.aps.copy_type  == 1) ? false : true; 
  uses_b = (hp.normal_form || hp.bps.copy_type  == 1) ? false : true;
  uses_c = true;
  uses_workspace = hp.aps.copy_type + hp.bps.copy_type + hp.normal_form == 0  ? false : true;
  uses_alpha  = true;
  uses_beta = dp.does_beta_c_inc;

}


public:
  //std::string kernelname = generickernelname;
  AlphaGenerator(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_, const derivedparams::DerivedParams & dp_, std::string & type_):
  basegen::BaseGenerator(hp_, gg_, dp_, type_)
  
  {
  
  
  }

virtual void setup() final override{
  
  set_usage();



}


private:
  
void append_group_allocation_string(std::stringstream & ss){
  if (hp.group_allocation == 1){
    ss << 
R"(
/* GROUP_ALLOCATION = 1 :  allocation is done column-by-column */
const unsigned group_id_vertical = group_id_xy % N_GROUPS_VERTICALLY;
const unsigned group_id_horizontal = group_id_xy / N_GROUPS_VERTICALLY;
)";
  }
  
  else if (hp.group_allocation == 2){
    ss << 
R"(
/* GROUP_ALLOCATION = 2 :  allocation is done row-by-row */
unsigned group_id_horizontal = group_id_xy % N_GROUPS_HORIZONTALLY;
unsigned group_id_vertical = group_id_xy / N_GROUPS_HORIZONTALLY;
)";
  }
  
  else if (hp.group_allocation == 3){
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
unsigned group_id_horizontal;
unsigned group_id_vertical;
unsigned wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*N_GROUPS_VERTICALLY);

  
if (group_id_xy < (N_GROUPS_HORIZONTALLY - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_VERTICALLY){
group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
group_id_vertical = (group_id_xy / SUPER_COLUMN_WIDTH) % N_GROUPS_VERTICALLY;
}
else{
group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % LAST_SUPER_COLUMN_WIDTH;
group_id_vertical = (group_id_xy  - (N_GROUPS_HORIZONTALLY - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_VERTICALLY) / LAST_SUPER_COLUMN_WIDTH;
}
)";

  }
  
  else{
    std::stringstream err_ss;
    err_ss << "Invalid group_allocation parameter : " << hp.group_allocation << ". It should be one of 1/2/3.";
    throw tinygemm_error(err_ss.str());
  }
}

void append_super_column_width_defn(std::stringstream & ss){

  if (hp.group_allocation == 3){
  
    ss <<  "\n\n" << "/* This variable defines the width of super-columns (we have GROUP_ALLOCATION 3). It is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) */\n" << "#define SUPER_COLUMN_WIDTH " << dp.ga3_super_column_width;   
    
    ss << "\n/* LAST_SUPER_COLUMN_WIDTH : N_GROUPS_HORIZONTALLY % SUPER_COLUMN_WIDTH  */";
    ss << "\n#define LAST_SUPER_COLUMN_WIDTH " << dp.ga3_last_super_column_width;
    
  }
}

void append_split_on_k_vardecl_write_string(std::stringstream & ss){
  if (dp.split_on_k != 0){
    ss << 
R"(
/* the following variables are used in implementing a basic atomic increment */
global TFLOAT * ptr_to_c_elm;  // with `restrict' is no faster
TFLOAT previous_value; )" << "\n" << dp.infa << " newVal;\n" << dp.infa << " prevVal;" << "\n\n";
  }
}

void append_loop_var_bound_incr(std::stringstream & ss, std::string varname, std::string bound_string, std::string increment_string){
  ss << "for (unsigned " << varname << " = 0; " << varname <<  " < " << bound_string  << "; " <<  increment_string << ")";
}


void append_a_load_for_perp(std::stringstream & ss){
  std::string bound_string = hp.load_to_lds_interwoven == 0 ? "MICRO_A_TILE_PERP_UNROLL" : "MACRO_TILE_HEIGHT";
  std::string increment_string = hp.load_to_lds_interwoven == 0 ? "++mu_perp_i" : "mu_perp_i += MACRO_TILE_HEIGHT/MICRO_A_TILE_PERP_UNROLL";
  append_loop_var_bound_incr(ss, "mu_perp_i", bound_string, increment_string);
}

void append_a_load_for_pll(std::stringstream & ss){
  std::string bound_string = hp.load_to_lds_interwoven == 0 ? "MICRO_A_TILE_PLL_UNROLL" : "UNROLL";
  std::string increment_string = hp.load_to_lds_interwoven == 0 ? "++mu_pll_i" : "mu_pll_i += UNROLL/MICRO_A_TILE_PLL_UNROLL";
  append_loop_var_bound_incr(ss, "mu_pll_i", bound_string, increment_string);
}

void append_b_load_for_perp(std::stringstream & ss){
  std::string bound_string = hp.load_to_lds_interwoven == 0 ? "MICRO_B_TILE_PERP_UNROLL" : "MACRO_TILE_WIDTH";
  std::string increment_string = hp.load_to_lds_interwoven == 0 ? "++mu_perp_i" : "mu_perp_i += MACRO_TILE_WIDTH/MICRO_B_TILE_PERP_UNROLL";
  append_loop_var_bound_incr(ss, "mu_perp_i", bound_string, increment_string);
}

void append_b_load_for_pll(std::stringstream & ss){
  std::string bound_string = hp.load_to_lds_interwoven == 0 ? "MICRO_B_TILE_PLL_UNROLL" : "UNROLL";
  std::string increment_string = hp.load_to_lds_interwoven == 0 ? "++mu_pll_i" : "mu_pll_i += UNROLL/MICRO_B_TILE_PLL_UNROLL";
  append_loop_var_bound_incr(ss, "mu_pll_i", bound_string, increment_string);
}


void append_final_write_element(std::stringstream & ss, unsigned atomic_increment, unsigned with_beta_scaling, unsigned with_alpha_increment){
  
  ss << "\nindex = ROW_STRIDE_C*(write_start_row + row) + COL_STRIDE_C*(write_start_col + col);\n";
  /* beta string */
  ss << (with_beta_scaling == 0 ? "" : "c[index] *= beta;\n");
  if (with_alpha_increment != 0){
    ss << "\n";
    if (atomic_increment == 0){
      ss << "c[index] += " + dp.alpha_scaled + ";\n"; 
    }
    
    else{
      ss  
      << "ptr_to_c_elm = c + index;\n" 
      << "do {\n"
      << "previous_value = *ptr_to_c_elm;\n" 
      << "prevVal = as_" << dp.infa << "(previous_value);\n"
      << "newVal = as_" << dp.infa << "(" << dp.alpha_scaled << " + previous_value);\n"
      << "} while (" << dp.fati << "(( __global " << dp.infa << "*)(ptr_to_c_elm), prevVal, newVal) != prevVal);";        
    }
  }
}

void append_for_loops_for_c_write_open(std::stringstream & ss){
  
  ss << "\n/* loops for writing to c */\n" << dp.pragma_unroll_string;
  append_loop_var_bound_incr(ss, "row", 
  hp.c_micro_tiles_interwoven == 0 ? "MICRO_TILE_HEIGHT" : "MACRO_TILE_HEIGHT", 
  hp.c_micro_tiles_interwoven == 0 ? "++row" : "row += N_MICRO_TILES_VERTICALLY");
  ss << " {\n" << dp.pragma_unroll_string;
  append_loop_var_bound_incr(ss, "col",
  hp.c_micro_tiles_interwoven == 0 ? "MICRO_TILE_WIDTH" : "MACRO_TILE_WIDTH", 
  hp.c_micro_tiles_interwoven == 0 ? "++col" : "col += N_MICRO_TILES_HORIZONTALLY");
  ss << " {\n";
}
  
void append_for_loops_for_c_write_close(std::stringstream & ss){
  ss << "\n}\n}\n";
}

void append_check_wrapped_if_clause_open(std::stringstream & ss){
  ss << 
R"(
/* catching the write cases for lower(l), right(r) and lr-corner tiles */
if (
((write_start_col + col >= MACRO_TILE_WIDTH*(N_GROUPS_HORIZONTALLY - 1)) && group_id_vertical   != (N_GROUPS_VERTICALLY   - 1 )) ||
((write_start_row + row >= MACRO_TILE_HEIGHT*(N_GROUPS_VERTICALLY - 1 )) && group_id_horizontal != (N_GROUPS_HORIZONTALLY - 1 )) ||
(
group_id_vertical == (N_GROUPS_VERTICALLY-1)     && 
group_id_horizontal == (N_GROUPS_HORIZONTALLY-1) && 
write_start_col + col >= MACRO_TILE_WIDTH*(N_GROUPS_HORIZONTALLY - 1) && 
write_start_row + row >= MACRO_TILE_HEIGHT*(N_GROUPS_VERTICALLY - 1)
)){
)";

}

void append_check_wrapped_if_clause_close(std::stringstream & ss){
  ss << "\n}";
}

void append_checked_wrapped_loops_from_bools(std::stringstream & ss, unsigned with_check, unsigned atomic_increment, unsigned with_beta_scaling, unsigned with_alpha_increment){
  
  append_for_loops_for_c_write_open(ss);
  if (with_check != 0){
    append_check_wrapped_if_clause_open(ss);
    append_final_write_element(ss, atomic_increment, with_beta_scaling, with_alpha_increment);
    append_check_wrapped_if_clause_close(ss);
  }
  
  else{
    append_final_write_element(ss, atomic_increment, with_beta_scaling, with_alpha_increment);
  }
  append_for_loops_for_c_write_close(ss);  
}
  

void append_final_write_loops(std::stringstream & ss, unsigned with_check){
  if (dp.split_on_k == 0){
    append_checked_wrapped_loops_from_bools(ss, with_check, 0, 1, 1);
  }
  
  else{
    append_checked_wrapped_loops_from_bools(ss, with_check, 1, 0, 1);
  }
}


void append_final_write_loops_no_check(std::stringstream & ss){
  append_final_write_loops(ss, 0);
}

void append_final_write_loops_with_check(std::stringstream & ss){
  append_final_write_loops(ss, 1);
}

void append_k_remaining_string(std::stringstream & ss){
  ss << "\nunsigned k_remaining = " << dp.effective_k_varies_string <<  " % UNROLL;";
}

void append_worktime_increment_ab(std::stringstream & ss, unsigned final_unroll){
  if (final_unroll == 0){
    std::string n_jumps_string = dp.split_on_k == 0 ? "UNROLL" : "G_UNROLL";
    ss << "a += " << dp.adps.main_pll_unroll_stride << "*" << n_jumps_string << ";\n",
    ss << "b += " << dp.bdps.main_pll_unroll_stride << "*" << n_jumps_string << ";\n";
  }
}

/* simple for loops. Could consider unrolling like Cobalt, but for the moment I use the optional pragma unroll */
void append_load_ab_into_LDS_string(std::stringstream & ss, unsigned final_unroll, unsigned special_first_unroll){
  
  if (final_unroll != 0 && special_first_unroll != 0){
    throw tinygemm_error("From get_load_ab_into_LDS_string > It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out");
  }
  
  if (hp.normal_form == 1 && (final_unroll != 0 || special_first_unroll != 0)){
    throw tinygemm_error("normal form should not have final or special first unrolls, from alphagenerator");
  }

  std::string a_value_to_get;
  std::string b_value_to_get;
  std::string a_comment;
  std::string b_comment;
  
  if (final_unroll == 1){
    a_value_to_get = "(a_offset_pll_unroll + mu_pll_i) < k_remaining ? a[mu_pll_i*COL_STRIDE_A + mu_perp_i*ROW_STRIDE_A] : 0;";
    b_value_to_get = "(b_offset_pll_unroll + mu_pll_i) < k_remaining ? b[mu_pll_i*ROW_STRIDE_B + mu_perp_i*COL_STRIDE_B] : 0;";
    a_comment =  "/* load final bit of data from a into LDS, less than a full unroll */";
    b_comment =  "/* load final bit of data from b into LDS, less than a full unroll */";
  }
  
  else if (special_first_unroll == 1){
    a_value_to_get = "(a_offset_pll_unroll + mu_pll_i) >= unroll_offset ? a[mu_pll_i*COL_STRIDE_A + mu_perp_i*ROW_STRIDE_A] : 0;";
    b_value_to_get = "(b_offset_pll_unroll + mu_pll_i) >= unroll_offset ? b[mu_pll_i*ROW_STRIDE_B + mu_perp_i*COL_STRIDE_B] : 0;";
    a_comment =  "/* load first bit of data from a into LDS, ignoring the prepended values (less than a full unroll)  */";
    b_comment =  "/* load first bit of data from b into LDS, ignoring the prepended values (less than a full unroll) */";
  }
  
  else{
    
    std::stringstream a_value_to_get_ss;
    a_value_to_get_ss << "a[mu_pll_i*" << dp.adps.main_pll_unroll_stride << " + mu_perp_i*" << dp.adps.main_perp_unroll_stride << "];";
    a_value_to_get = a_value_to_get_ss.str();
    
    std::stringstream b_value_to_get_ss;
    b_value_to_get_ss << "b[mu_pll_i*" << dp.bdps.main_pll_unroll_stride << " + mu_perp_i*" << dp.bdps.main_perp_unroll_stride << "];";
    b_value_to_get = b_value_to_get_ss.str();
    
    a_comment =  "/* load data from a into LDS */";
    b_comment =  "/* load data from b into LDS */";
  }
  
  ss << "\n" << a_comment << "\n"
  << dp.pragma_unroll_string;
  append_a_load_for_perp(ss);
  ss << " {\n" 
  << dp.pragma_unroll_string;
  append_a_load_for_pll(ss);
  ss << " {\n"
  << "localA[MACRO_TILE_HEIGHT_AND_PAD*(a_offset_pll_unroll + mu_pll_i) + (a_offset_perp_unroll + mu_perp_i)] = \n" 
  << a_value_to_get << "\n" 
  <<  "}\n" 
  <<  "}\n";


  ss 
  << "\n" << b_comment << "\n"
  << dp.pragma_unroll_string;
  append_b_load_for_perp(ss);
  ss << " {\n" 
  << dp.pragma_unroll_string;
  append_b_load_for_pll(ss);
  ss << " {\n"
  << "localB[MACRO_TILE_WIDTH_AND_PAD*(b_offset_pll_unroll + mu_pll_i) + (b_offset_perp_unroll + mu_perp_i)] = \n" 
  << b_value_to_get << "\n" 
  <<  "}\n" 
  <<  "}\n";

  ss << "\n";
  append_worktime_increment_ab(ss, final_unroll);
}

std::string get_c_work_item_vertical_next(){
  return hp.c_micro_tiles_interwoven != 0 ? "1" : "MICRO_TILE_HEIGHT";
}


std::string get_c_work_item_horizontal_next(){
  return hp.c_micro_tiles_interwoven != 0 ? "1" : "MICRO_TILE_WIDTH";
}
    
void append_relocate_lAlB_string(std::stringstream & ss){ 
  ss << 
  "\n" << "lA = localA + micro_id_vertical*" << get_c_work_item_vertical_next() << ";" <<  
  "\n" << "lB = localB + micro_id_horizontal*" << get_c_work_item_horizontal_next() << ";" << "\n";
}


/* We previously had a variable UNROLL_the_math_section = False. */
/* Experiments with UNROLL_the_math_section suggest that it's a bad idea. */
void append_math_section(std::stringstream & ss, unsigned use_k_remaining){
  
  std::string number_of_unrolls = use_k_remaining == 0 ? "UNROLL" : "k_remaining";
  ss << "\nfor (unsigned u = 0; u < " << number_of_unrolls << "; ++u){\n";
  append_load_load_string(ss);
  ss << "\n";
  append_compute_string(ss);  
  ss << "\n}\n";
}

void append_relocate_load_math_string(std::stringstream & ss, unsigned final_unroll, unsigned special_first_unroll){
  if (final_unroll != 0 && special_first_unroll != 0){
    throw tinygemm_error("From get_relocate_load_math_string : It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out");
  }

  append_load_ab_into_LDS_string(ss, final_unroll, special_first_unroll);
  ss << 
R"(
/* make sure all loads from LDS memory have completed */
barrier(CLK_LOCAL_MEM_FENCE); )";

  append_relocate_lAlB_string(ss);
  append_math_section(ss, final_unroll);
  ss << 
R"(
/* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
barrier(CLK_LOCAL_MEM_FENCE); )";
}
  
void append_final_unroll_string(std::stringstream & ss){
  
  if (dp.split_on_k == 0){
    ss << "\n";
    append_relocate_load_math_string(ss, 1, 0);
    ss << "\n";
  }
  else{
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

void append_first_unroll_block(std::stringstream & ss){
  if (hp.unroll_for_offset != 0){
    ss << "\n\n/* This is where the first unroll will be performed. Identical to what is in the main while, but with zero buffering.  */";
    if (dp.split_on_k == 0){
      ss << "\n";
      append_relocate_load_math_string(ss, 0, 1); 
      ss << "\n--n_unrolls_remaining;\n";
    }
    else{
      ss << "\nif (group_id_z == 0){\n";
      append_relocate_load_math_string(ss, 0, 1);
      ss << "\n--n_unrolls_remaining;\n}";
    }
  }
}

void append_compute_string(std::stringstream & ss){
  ss 
  << dp.pragma_unroll_string 
  << "for (unsigned row = 0; row < MICRO_TILE_HEIGHT; ++row){\n"
    
  << dp.pragma_unroll_string 
  << "for (unsigned col = 0; col < MICRO_TILE_WIDTH; ++col){\n" 

  << "//mad(rA[row],rB[col],rC[row][col]);" 
  << "\n//can the compiler change these unsigneds to chars? if not, maybe try. " 
  << "\n//That said, it's going to be unrolled anyway, so not worth it." 
  << "\nrC[row][col] += rA[row]*rB[col];   \n}\n}\n";
} 

/* This returns the section which makes the within work-group adjustments to a, b to
 * put a work item in the correct position to load its first element from global
 * if the load tiles are interlaced (ala cobalt), this final offset is just 1
 * row or column. If the tiles are not interlaced, this final offset is the width
 * or height of the load tile. */    
void append_micro_offset_string(std::stringstream & ss){
  
    
    
    std::string str_a_n_pll(""); 
    std::string str_a_n_perp(""); 
    std::string str_b_n_pll("");
    std::string str_b_n_perp("");
    
    if (hp.load_to_lds_interwoven == 0){
      str_a_n_pll = "MICRO_A_TILE_PLL_UNROLL *";
      str_a_n_perp = "MICRO_A_TILE_PERP_UNROLL *";
      str_b_n_pll = "MICRO_B_TILE_PLL_UNROLL *";
      str_b_n_perp = "MICRO_B_TILE_PERP_UNROLL *";
    }

    ss << "\n/* make the micro adjustments (A) for the thread, getting ready to load */\n";
    ss << "const unsigned a_offset_pll_unroll = " << str_a_n_pll << " pll_unroll_a_load_id;\n";
    ss << "const unsigned a_offset_perp_unroll = " << str_a_n_perp <<  " perp_unroll_a_load_id;\n";
  
    ss << "a += " << dp.adps.main_pll_unroll_stride << " * a_offset_pll_unroll;\n";
    ss << "a += " << dp.adps.main_perp_unroll_stride << " * a_offset_perp_unroll;\n\n";
  
    
    ss << "/* make the micro adjustments (B) for the thread, getting ready to load */\n";
    ss << "const unsigned b_offset_pll_unroll = " << str_b_n_pll << " pll_unroll_b_load_id;\n";
    ss << "const unsigned b_offset_perp_unroll = " << str_b_n_perp << " perp_unroll_b_load_id;\n";

    ss << "b += " << dp.bdps.main_pll_unroll_stride << " * b_offset_pll_unroll;\n";
    ss << "b += " << dp.bdps.main_perp_unroll_stride << " * b_offset_perp_unroll;\n";

}

void append_load_load_string(std::stringstream & ss){
  ss << "\n" << dp.pragma_unroll_string << "for (unsigned i = 0; i < MICRO_TILE_HEIGHT; ++i){\nrA[i] = lA[" << dp.adps.main_strided_i << "];\n}\n";
  ss << "lA += MACRO_TILE_HEIGHT_AND_PAD;\n\n" << dp.pragma_unroll_string;
  ss << "for (unsigned i = 0; i < MICRO_TILE_WIDTH; ++i){\nrB[i] = lB[" << dp.bdps.main_strided_i << "];\n}\n";
  ss << "lB += MACRO_TILE_WIDTH_AND_PAD;";
}

void append_localA_localB_decl_string(std::stringstream & ss){
  ss << "__local TFLOAT localA[N_ELEMENTS_IN_PADDED_A_UNROLL];\n__local TFLOAT localB[N_ELEMENTS_IN_PADDED_B_UNROLL];\n";
}


void append_preshift_defns(std::stringstream & ss){
  
  
  if (dp.use_edge_trick != 0){
    ss << "\n" << 
    "/* 1 + (__M__ - 1) % MACRO_TILE_HEIGHT  */ \n" << 
    "#define PRESHIFT_BOTTOMMOST_TILE_HEIGHT " << dp.adps.main_preshift_final_tile << " // somewhere in 1 ... MACRO_TILE_HEIGHT\n" << 
    "/* 1 + (__N__ - 1) % MACRO_TILE_WIDTH */ \n" << 
    "#define PRESHIFT_RIGHTMOST_TILE_WIDTH " << dp.bdps.main_preshift_final_tile << " // somewhere in 1 ... MACRO_TILE_WIDTH\n";
  }
}

void append_special_case_edge_trick_string(std::stringstream & ss){
  ss << "\n";
  //TODO : if normal form, a b in copy will be buffered.
  if (hp.normal_form == 0 && dp.use_edge_trick != 0){
    ss <<
R"(/* Special case of the tile being on far right : pull the tile to the left just enough so that it doesn't overflow C */
if (group_id_horizontal == N_GROUPS_HORIZONTALLY - 1){
macro_tile_start_col_in_c -= (MACRO_TILE_WIDTH - PRESHIFT_RIGHTMOST_TILE_WIDTH);
}

/* Special case of the tile being on the bottom : pull the tile up just enough so that it doesn't overflow C */    
if (group_id_vertical == N_GROUPS_VERTICALLY - 1){
macro_tile_start_row_in_c -= (MACRO_TILE_HEIGHT - PRESHIFT_BOTTOMMOST_TILE_HEIGHT);
}
)";

  }
}

void append_group_allocation_defn_string(std::stringstream & ss){
  ss << "#define GROUP_ALLOCATION " << hp.group_allocation;
  if (hp.group_allocation == 3){
    ss << "/* this variable is declared because we have GROUP_ALLOCATION type 3. */\n";
    ss << "/* It should define how many workgroups we expect to have active simulantaneuosly. */\n";
    ss << "#define N_TARGET_ACTIVE_WORKGROUPS " << hp.n_target_active_workgroups;
  }
}

void append_ngroups_grid_string(std::stringstream & ss){
  ss << "\n/* the number of work groups vertically and horizontally. */\n/* note that this ignores n_work_items_per_c_elm, so that only one workgroup per c cell is used in computing this */ ";
  
  ss << "\n"
  << "/* number of groups vertically : __M__ / MACRO_TILE_HEIGHT";
  if (dp.use_edge_trick == 1){
    ss << " + (PRESHIFT_BOTTOMMOST_TILE_HEIGHT != MACRO_TILE_HEIGHT)";
  } 
  ss << " */" << "\n"
  << "#define N_GROUPS_VERTICALLY " <<  dp.adps.main_n_groups << "\n"
  << "/* number of groups horizontally : __N__ / MACRO_TILE_WIDTH";
  if (dp.use_edge_trick == 1){
    ss << " + (PRESHIFT_RIGHTMOST_TILE_WIDTH != MACRO_TILE_WIDTH)";
  }
  ss << "*/" << "\n"
  << "#define N_GROUPS_HORIZONTALLY " <<  dp.bdps.main_n_groups << "\n";

}


void append_final_write_all(std::stringstream & ss){
  
  if (dp.use_edge_trick == 0){
    ss << "\n";
    append_final_write_loops_no_check(ss);
  }
  
  else{
    ss << 
R"(
/* the case where this is not an edge tile : will write to all cells */
if ((group_id_horizontal != N_GROUPS_HORIZONTALLY - 1 || PRESHIFT_RIGHTMOST_TILE_WIDTH == MACRO_TILE_WIDTH) 
&& (group_id_vertical != N_GROUPS_VERTICALLY - 1 || PRESHIFT_BOTTOMMOST_TILE_HEIGHT == MACRO_TILE_HEIGHT)){
)";
    append_final_write_loops_no_check(ss);
    ss << "\n}\n\nelse{";
    append_final_write_loops_with_check(ss);
    ss << "\n}";
  }
}

void append_split_on_k_ab_offset_adjustment_string(std::stringstream & ss){
  ss << "\n";
  if (dp.split_on_k != 0){
    ss << 
R"(
/* a,b are pointing to the top left of the region required by the macro tile, but this work group  */
/* might not process the whole of a and b. We now turn 90 and shift pointers a,b to the start for this wg */)";

    if (hp.normal_form == 0){
      ss <<
R"(
a += UNROLL*group_id_z*COL_STRIDE_A;
b += UNROLL*group_id_z*ROW_STRIDE_B;
)";
    }
    
    else{
      ss <<
R"(
a += group_id_z*N_ELEMENTS_IN_A_UNROLL;
b += group_id_z*N_ELEMENTS_IN_B_UNROLL;
)";

      
    }
  }
}

void append_k_unroll_offset_initial_string(std::stringstream & ss){
  
  
  if (hp.normal_form == 0 && hp.unroll_for_offset != 0){
    ss <<
R"(
/* this additional offset of a and b appears because UNROLL_FOR_OFFSET is 1 */
unsigned unroll_offset = (3*group_id_vertical + 11*group_id_vertical)%UNROLL;
unsigned k_plus_offset = __K__ + unroll_offset;
a -= unroll_offset*COL_STRIDE_A;
b -= unroll_offset*ROW_STRIDE_B;
)";
  }
}

void append_split_on_k_defns_string(std::stringstream & ss){
  if (dp.split_on_k != 0){
    ss << 
R"(
/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL )" << hp.n_work_items_per_c_elm*hp.unroll << " // N_WORK_ITEMS_PER_C_ELM*UNROLL";
  }
}

void append_group_id_defns(std::stringstream & ss){
  if (dp.split_on_k == 0){
    ss << "\nconst unsigned group_id_xy = get_group_id(0);\n";
  }
  else{
    ss << 
R"(
const unsigned group_id = get_group_id(0);
const unsigned group_id_xy = group_id / N_WORK_ITEMS_PER_C_ELM;
const unsigned group_id_z = group_id % N_WORK_ITEMS_PER_C_ELM;
)";

  }
}

  
void append_stride_defn(std::stringstream & ss, char LETTER, unsigned ldx,  unsigned transposed){
  
  unsigned transposed_xor_is_col_major = (transposed + gg.isColMajor) % 2;
  
  std::stringstream ldx_string_ss;
  ldx_string_ss << "LD" << LETTER;
  std::string ldx_string = ldx_string_ss.str();
  
  
  //TODO merge the four message functions, there is redundancy
  ss << "\n/* To move from " << LETTER << "[row][col] to " << LETTER << "[row+1][col], how much should the pointer increment? As we have " << LETTER << "_TRANSPOSED = " << transposed << " and IS_COL_MAJOR = " << gg.isColMajor << ", this is " << (transposed_xor_is_col_major == 1 ? "1" : ldx_string) << " */\n";
  
  ss << "#define ROW_STRIDE_" << LETTER << " " << (transposed_xor_is_col_major == 1 ? 1 : ldx) << "\n";

  ss << "/* To move from " << LETTER << "[row][col] to " << LETTER << "[row][col+1], how much should the pointer increment? As we have " << LETTER << "_TRANSPOSED = " << transposed << " and IS_COL_MAJOR = " << gg.isColMajor << ", this is " << (transposed_xor_is_col_major == 1 ? ldx_string : "1") << " */\n";

  ss <<"#define COL_STRIDE_" << LETTER << " " << (transposed_xor_is_col_major == 1 ? ldx : 1) << "\n";
  
}

void append_stride_defns(std::stringstream & ss){
  if (hp.normal_form == 0){
    append_stride_defn(ss, 'A', dp.adps.main_effective_ldx, gg.tA);
    append_stride_defn(ss, 'B', dp.bdps.main_effective_ldx, gg.tB);
  }
  append_stride_defn(ss, 'C', gg.ldc, gg.tC);  
}

void append_rA_rB_decl_string(std::stringstream & ss){
  ss << "TFLOAT rA[MICRO_TILE_HEIGHT];\nTFLOAT rB[MICRO_TILE_WIDTH];\n";
}

void append_n_unrolls_remaining_string(std::stringstream & ss){
  std::string k_effective_mod_G_UNROLL = dp.effective_k_varies_string + " % G_UNROLL";
  std::string k_effective_div_G_UNROLL = dp.effective_k_varies_string + " / G_UNROLL";
  std::string k_effective_div_UNROLL = dp.effective_k_varies_string + " / UNROLL";
  
  if (dp.split_on_k == 0){
    ss << "\nint n_unrolls_remaining = " << k_effective_div_UNROLL << ";";
  }
  
  else{    
    ss << "\n/* a certain number of work groups process one more unroll. Note that with UFO = 1, this depends on column */";
    ss << "\nconst int n_work_groups_with_1_more = (" << k_effective_mod_G_UNROLL << ") / UNROLL; \n";
    ss << "\n// branching between work groups : some wgs have 1 more unroll to process.\nint n_unrolls_remaining = (" << k_effective_div_G_UNROLL;
    ss << ") +  (group_id_z < n_work_groups_with_1_more);";
  }
}

void append_what_string(std::stringstream & ss){
  ss << "A:" << gg.tA << "B" << gg.tB << "C" << gg.tC << "f" << gg.derived.float_size_bits;
}

void append_kernel_name(std::stringstream & ss){
  ss << kernelname;
}


   
  

void append_initial_a_offset(std::stringstream & ss){
  if (hp.aps.copy_type == 1){
    ss << "/* a from workspace */\n";
    ss << "const TFLOAT * restrict a = workspace + workspace_offset;\n";
  }
  
  else{
    ss << "a += a_offset;\n";
  }

}


void append_initial_b_offset(std::stringstream & ss){

  if (hp.bps.copy_type == 1){
    ss << "/* b from workspace */\n";
    ss << "const TFLOAT * restrict b = workspace + workspace_offset + GLOBAL_OFFSET_B;\n";
  }
  
  else{
    ss << "b += b_offset;\n";
  }

}


void append_a_offset_string(std::stringstream & ss){

  ss << R"(


/* In OpenCL, host code does not have access to raw data pointers. */
/* Host code works with cl_mem objects, which encapsulate and hide raw points. */
/* For this reason, host code CANNOT simply increment pointers to data, */
/* as one can do with pointers for CPU gemm, or cublas gemm for that matter. */

c += c_offset;
)";
  
  append_initial_a_offset(ss);
  append_initial_b_offset(ss);
  
  ss << "\n\n\n";
  
}

  

void append_global_offset_b_workspace(std::stringstream & ss){
  if (hp.bps.copy_type == true){
    ss << "\n\n/* b is offset in workspace because a comes before it */\n#define GLOBAL_OFFSET_B " << dp.bdps.cw_global_offset;
  }
}




void append_ldx_definitions(std::stringstream & ss){

  if (hp.normal_form == 0){
    
    if (hp.aps.copy_type == 1){
      ss << "/* as per a in workspace */\n";
    }
    ss << "#define __LDA__ " << dp.adps.main_effective_ldx << "\n";
    
    if (hp.bps.copy_type == 1){  
      ss << "/* as per b in workspace */\n";
    }
    ss << "#define __LDB__ " << dp.bdps.main_effective_ldx << "\n";
  
  }
  
  ss << "#define __LDC__ " << gg.ldc << "\n";
}


void append_mnk_definitions(std::stringstream & ss){
  ss << "#define __M__ " << gg.m << "\n";
  ss << "#define __N__ " << gg.n << "\n";
  if (hp.normal_form == 0){
    ss << "#define __K__ " << gg.k << "\n";
  }
  else{
    ss << "#define __K_NORMAL_FORM__ " << dp.nf_k_normal_form << "\n";
  }
}


void append_transpose_definitions(std::stringstream & ss){
  ss << "#define IS_COL_MAJOR " << gg.isColMajor << "\n";
  ss << "#define C_TRANSPOSED " << gg.tC <<  "\n";
  if (hp.normal_form == 0){  
    ss << "#define A_TRANSPOSED " << gg.tA << "\n";
    ss << "#define B_TRANSPOSED " << gg.tB <<  "\n";

  }
}


void append_transpose_note(std::stringstream & ss){
  if (hp.normal_form == 0){
    ss << 
    R"(
/* A note on how transposes IS_COL_MAJOR, A_TRANSPOSED, B_TRANSPOSED and C_TRANSPOSED effect the kernel generated: */
/* very little. The only way they effect the kernel is through ROW_STRIDE_X, COL_STRIDE_X, for X in {A,B,C}. */
/* ROW_STRIDE_X and COL_STRIDE_X are either 1 of ldx, depending on xor(IS_COL_MAJOR, X_TRANSPOSED) */ 
/* the notation and comments which follow assume transposes are false, technically ROW_STRIDE_A should be ROW_STRIDE_OPA */
/* in short, to understand tinygemm kernels, it is enough to understand the non-transpose case, all other cases are minor modifications */
)";
  }
}

void append_move_to_top_corner_string(std::stringstream & ss){
  ss << "\n/* move to the top left corner of a (top left corner of b) of region required by the macro tile */";
  if (hp.normal_form == 0){
    ss << 
R"(
a += macro_tile_start_row_in_c*ROW_STRIDE_A;   
b += macro_tile_start_col_in_c*COL_STRIDE_B;
)";
  }
  else{
    ss << 
  R"(
a += macro_tile_start_row_in_c*__K_NORMAL_FORM__;   
b += macro_tile_start_col_in_c*__K_NORMAL_FORM__;
)";
  }
}


void append_load_within_tile_pattern_string(std::stringstream & ss){  
  
  ss << 
R"(
///* Define which rows or columns of A, B this thread will load from global into LDS (% / or / % ? looks like no difference ) */

const unsigned pll_unroll_a_load_id = local_id % N_MICRO_A_TILES_PLL_UNROLL;
const unsigned perp_unroll_a_load_id = local_id / N_MICRO_A_TILES_PLL_UNROLL;

const unsigned pll_unroll_b_load_id = local_id % N_MICRO_B_TILES_PLL_UNROLL;
const unsigned perp_unroll_b_load_id = local_id / N_MICRO_B_TILES_PLL_UNROLL;

/* flipping the order doesn't seem to make any difference */
//const unsigned pll_unroll_a_load_id = local_id / (N_WORK_ITEMS_PER_WORKGROUP / N_MICRO_A_TILES_PLL_UNROLL);
//const unsigned perp_unroll_a_load_id = local_id % (N_WORK_ITEMS_PER_WORKGROUP / N_MICRO_A_TILES_PLL_UNROLL);

//const unsigned pll_unroll_b_load_id = local_id / (N_WORK_ITEMS_PER_WORKGROUP / N_MICRO_B_TILES_PLL_UNROLL);
//const unsigned perp_unroll_b_load_id = local_id % (N_WORK_ITEMS_PER_WORKGROUP / N_MICRO_B_TILES_PLL_UNROLL);


)";
}


  
public:

/* the "main" kernel */
KernelString get_kernelstring(){
  
      
  std::stringstream ss;
  
  ss << genutil::get_time_string(type);

  ss <<  "\n\n" << genutil::get_what_string() << "\n";

  append_mnk_definitions(ss);

  append_ldx_definitions(ss);
  
  append_transpose_definitions(ss);

  ss << "#define TFLOAT  "  << dp.t_float << "\n";
  
  ss << "#define DOES_BETA_C_INC " << dp.does_beta_c_inc << "\n";
  ss << "#define DOES_ALPHA_A_B_INC 1" << "\n";
  
  append_transpose_note(ss);



  ss << 
R"(
/* TODO : remove final barrier, not nec. Check performance is not mysteriously hit! */
/* TODO : beta = 1 optimisation */
/* TODO : investigate mad. When should one use this instead of standard overloads, += and *. see tensile branch. 


)" << genutil::get_how_string() << R"(
/* Defines a tile shape. Each thread will process a tile of this shape from C (if N_WORK_ITEMS_PER_C_ELM > 1 the processing is shared with threads in other WGs)  */
)";
  ss << "#define MICRO_TILE_WIDTH " << hp.bps.micro_tile_length << "\n";
  ss << "#define MICRO_TILE_HEIGHT " << hp.aps.micro_tile_length << "\n";
  ss << "/* Area of C which a workgroup will process. Recall that a workgroup is made of several threads which share LDS memory */\n";
  ss << "#define MACRO_TILE_WIDTH " << hp.bps.macro_tile_length << "\n";
  ss << "#define MACRO_TILE_HEIGHT " << hp.aps.macro_tile_length << "\n";
  ss << "/* How much a workgroup load (global -> LDS) in the k-direction at each iteration of the outer-most loop */\n";
  ss << "#define UNROLL " << hp.unroll  << "\n";
  ss << "/* padding in LDS to avoid bank conflicts*/\n";
  ss << "#define PAD_LDS_A " << hp.aps.lds_pad_size << "\n";
  ss << "#define PAD_LDS_B " << hp.bps.lds_pad_size << "\n";
  ss << "/* whether or not this kernel uses the edge trick (see documentation : (TODO, currently internal AMD document)) */\n";
  ss << "/* this precompiler defn has no direct influence on the running the kernel, implementation already done in make_kernel.py */\n";
  ss << "#define EDGETRICK " << dp.use_edge_trick << "\n";
  ss << "/* the number of work items working on the same c element. if this is 1, there will be just one thread doing all k multiply-adds, */\n";
  ss << "/* otherwise if it is greater than 1, each thread will be computing ~ k / N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added at the end */ \n";
  ss << "#define N_WORK_ITEMS_PER_C_ELM " << hp.n_work_items_per_c_elm << "\n";
  ss << "/* whether or not to shimmy the starting k, in an attempt to avoid cache line overuse for cases where lda/ldb are powers of 2 */\n";
  ss << "/* if 0, no shimmying. if 1, instead of starting at k = 0 workgroups start at some negative offset dependent on work group id */\n";
  ss << "/* in the same way as the final unroll populates LDS with zeros in k mod UNROLL != 0, the initial negative indices here populate with 0 */\n";
  if (hp.normal_form == 0){
    ss << "#define UNROLL_FOR_OFFSET " << hp.unroll_for_offset << "\n";
  }
  ss << 
R"(
/* define the way in which work groups are assigned to tiles */
/* 1 : column-by-column
 * 2 : row-by-row 
 * 3 : by rows within super-column  */
)";

  append_group_allocation_defn_string(ss);
  
  ss << 
R"(
/* Whether the load tiles are long in the direction of unroll (1) or perpendicular to the direction of unroll (0) */
/* Note : if the load tiles are long in the direction of unroll, the destination tile in LDS is NOT contiguous,  */
/* in other words, unroll tiles are stored with the the direction perpendicular to unroll as the fastest changing index */
/* We include these parameters here as pre-processor variables, but the loading micro-tile shapes are set in make_kernel.py */
)";

  ss << "#define WORK_ITEM_LOAD_A_PLL_TO_UNROLL " << hp.aps.load_pll_to_unroll << "\n"; 
  ss << "#define WORK_ITEM_LOAD_B_PLL_TO_UNROLL " << hp.aps.load_pll_to_unroll << "\n";
  ss << "/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if the load tiles are truly contiguous tiles of A/B (0) */\n";
  ss << "/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */\n";
  ss << "#define LOAD_TO_LDS_INTERWOVEN " << hp.load_to_lds_interwoven << "\n";
    

  

  ss << "/* Whether the micro tile being processed by a compute item is interwoven with other micro tiles (ala Cobalt, (1)) or if the micro tiles are contiguous in C */\n";
  ss << "/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */\n";
  ss << "#define C_MICRO_TILES_INTERWOVEN " << hp.c_micro_tiles_interwoven << "\n";
  ss << "/* Whether to use the unroll pragma to encourage the compiler to unroll certain loops */\n";
  ss << "/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */\n";
  ss << "#define PRAGMA_UNROLL_FORLOOPS " << hp.unroll_pragma << "\n";
  ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into registers, as compared to doing the math. */\n";
  ss << "/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */\n";
  ss << "#define N_PREFETCH_FOR_REGISTER_LOAD " << 0 << "\n";
  ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into LDS, as compared to the unroll loop */\n";
  ss << "/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */\n";
  ss << "#define N_PREFETCH_FOR_LDS_LOAD " << 0 << "\n";
  ss << "\n\n\n\n" << genutil::get_derived_string() << "\n";
  ss << R"(/*  <+_+>
 *           <+_+>
 *      <+_+>
 * TODO : rerereconsider assignment of workitems to load and math regions, there may be some sweet overlap spot where values automatically in registers for math (?) */
)";
  

  ss << "\n";
  ss << "#define MACRO_TILE_AREA "<< dp.macro_tile_area <<"  // MACRO_TILE_WIDTH*MACRO_TILE_HEIGHT\n";
  ss << "#define MICRO_TILE_AREA "<< dp.micro_tile_area <<" // MICRO_TILE_WIDTH * MICRO_TILE_HEIGHT\n";
  ss << "#define N_WORK_ITEMS_PER_WORKGROUP  "<< dp.main_n_work_items_per_workgroup <<" // MACRO_TILE_AREA / MICRO_TILE_AREA\n";
  ss << "#define MACRO_TILE_HEIGHT_AND_PAD "<< dp.adps.main_macro_tile_length_and_pad <<" // MACRO_TILE_HEIGHT + PAD_LDS_A\n";
  ss << "#define MACRO_TILE_WIDTH_AND_PAD "<< dp.bdps.main_macro_tile_length_and_pad <<" // MACRO_TILE_WIDTH + PAD_LDS_B\n";
  ss << "#define N_ELEMENTS_IN_A_UNROLL "<< dp.adps.n_elements_in_unroll <<" // MACRO_TILE_HEIGHT * UNROLL\n";
  ss << "#define N_ELEMENTS_IN_B_UNROLL "<< dp.bdps.n_elements_in_unroll <<" // MACRO_TILE_WIDTH * UNROLL\n";
  ss << "#define N_ELEMENTS_IN_PADDED_A_UNROLL "<< dp.adps.main_n_elements_in_padded_unroll <<" // MACRO_TILE_HEIGHT_AND_PAD * UNROLL\n";
  ss << "#define N_ELEMENTS_IN_PADDED_B_UNROLL "<< dp.bdps.main_n_elements_in_padded_unroll <<" // MACRO_TILE_WIDTH_AND_PAD * UNROLL\n";
  ss << "#define N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM "<< dp.adps.main_n_elements_to_load_per_workitem <<" // N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP\n";
  ss << "#define N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM "<< dp.bdps.main_n_elements_to_load_per_workitem <<" // N_ELEMENTS_IN_B_UNROLL / N_WORK_ITEMS_PER_WORKGROUP\n";
  ss << "#define N_MICRO_TILES_VERTICALLY "<< dp.adps.main_n_micro_in_macro <<" // MACRO_TILE_HEIGHT / MICRO_TILE_HEIGHT\n";
  ss << "#define N_MICRO_TILES_HORIZONTALLY "<< dp.bdps.main_n_micro_in_macro <<" // MACRO_TILE_WIDTH / MICRO_TILE_WIDTH\n";

  ss << "\n/* MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM */\n";  
  ss << "/* The dimensions of a tile in A loaded by a work item.  */\n";
  ss << "#define MICRO_A_TILE_PLL_UNROLL " << dp.adps.main_micro_tile_pll_unroll << " // size of the loaded tile, pll to unroll\n";
  ss << "#define MICRO_A_TILE_PERP_UNROLL " << dp.adps.main_micro_tile_perp_unroll << "\n";
  ss << "#define N_MICRO_A_TILES_PLL_UNROLL " << dp.adps.main_n_micro_tiles_pll_unroll << " // UNROLL / MICRO_A_TILE_PLL_UNROLL\n""";
  ss << "\n/*  MICRO_B_TILE_PLL_UNROLL * MICRO_B_TILE_PERP_UNROLL = N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM */\n";
  ss << "/* The dimensions of a tile in B read by a work item */\n";
  ss << "#define MICRO_B_TILE_PLL_UNROLL " << dp.bdps.main_micro_tile_pll_unroll << "\n";
  ss << "#define MICRO_B_TILE_PERP_UNROLL " << dp.bdps.main_micro_tile_perp_unroll << "\n";
  ss << "#define N_MICRO_B_TILES_PLL_UNROLL " << dp.bdps.main_n_micro_tiles_pll_unroll << " // UNROLL / MICRO_B_TILE_PLL_UNROLL\n";
  
  ss << "\n/* two more parameters, which do dot have an effect the running of this kernel (used in enqueuing) */\n";
  ss << "/* the total number of work groups this kernel will use (recall m,n,k are fixed) */ \n";
  ss << "/* N_WORK_ITEMS_PER_C_ELM * ((__M__/MACRO_TILE_HEIGHT) + (__M__%MACRO_TILE_HEIGHT != 0)) * ((__N__/MACRO_TILE_WIDTH) + (__N__%MACRO_TILE_WIDTH != 0)) */ \n";
  ss << "#define N_WORK_GROUPS " << dp.main_n_work_groups << "\n";
  ss << "/* the global work size, ie the total mumber of work items (threads) which will run */ \n";
  ss << "/* N_WORK_GROUPS * N_WORK_ITEMS_PER_WORKGROUP */ \n";
  ss << "#define GLOBAL_WORK_SIZE " << dp.main_global_work_size << "\n";
  ss << "\n";
  
  append_stride_defns(ss);
  append_preshift_defns(ss);
  append_ngroups_grid_string(ss);
  append_split_on_k_defns_string(ss);
  append_super_column_width_defn(ss);
  append_global_offset_b_workspace(ss);
  
  ss << "\n\n\n__attribute__((reqd_work_group_size(" << dp.main_n_work_items_per_workgroup << ",1, 1)))\n";
  ss << "__kernel void ";
  append_kernel_name(ss);
  append_parameter_list_from_usage(ss);
  
  ss << "\n{\n\n";

  append_a_offset_string(ss);

  append_group_id_defns(ss);
  ss << "const unsigned local_id = get_local_id(0);\n";
  append_group_allocation_string(ss);
    
  ss << 
R"(
unsigned macro_tile_start_row_in_c = group_id_vertical*MACRO_TILE_HEIGHT;
unsigned macro_tile_start_col_in_c = group_id_horizontal*MACRO_TILE_WIDTH;  

)";

  append_special_case_edge_trick_string(ss);

  append_move_to_top_corner_string(ss);
  
  append_split_on_k_ab_offset_adjustment_string(ss);

  append_k_unroll_offset_initial_string(ss);

  append_load_within_tile_pattern_string(ss);

  append_micro_offset_string(ss);
  
  ss << 
R"(

/* Define which part of the C macro-tile this thread will process (% / or / % ? doesn't seem to make much difference) */

const unsigned micro_id_vertical = local_id % N_MICRO_TILES_VERTICALLY;
const unsigned micro_id_horizontal = local_id / N_MICRO_TILES_VERTICALLY;

//const unsigned micro_id_vertical = local_id / N_MICRO_TILES_HORIZONTALLY;
//const unsigned micro_id_horizontal = local_id % N_MICRO_TILES_HORIZONTALLY;

  
)";

  ss << "/* LDS memory */\n";
  append_localA_localB_decl_string(ss);
  ss << "/* register memory */\n    TFLOAT rC[MICRO_TILE_HEIGHT][MICRO_TILE_WIDTH] = {{0.}};\n";
  append_rA_rB_decl_string(ss);
  ss << 
R"(
/* jumping pointers to locate the LDS to load into register memory */
__local const TFLOAT * lA;
__local const TFLOAT * lB;

)";

  append_n_unrolls_remaining_string(ss);
  append_first_unroll_block(ss);

  ss << "\n\nwhile (n_unrolls_remaining > 0){\n";
  append_relocate_load_math_string(ss, 0,0);
  ss << "\n--n_unrolls_remaining;\n}\n";
  
  if (dp.needs_final_fractional_unroll == 1){
    ss << "\n/* *********** processing the tail *************** */\n";
    append_k_remaining_string(ss);
    append_final_unroll_string(ss);
    ss << "\n/* *********************************************** */\n\n";
  }
  
  
  ss << "\n\n";
  ss << "const unsigned write_start_row = macro_tile_start_row_in_c + micro_id_vertical*" << get_c_work_item_vertical_next() << ";\n";
  ss << "const unsigned write_start_col = macro_tile_start_col_in_c + micro_id_horizontal*" << get_c_work_item_horizontal_next() << ";\n";  
  ss << "unsigned index;\n";
  
  append_split_on_k_vardecl_write_string(ss);
  append_final_write_all(ss);
  ss << "\n}\n";


  


  
  return { {uses_a, uses_b, uses_c, uses_workspace, uses_alpha, uses_beta } , ss.str(), kernelname, dp.main_global_work_size, dp.main_n_work_items_per_workgroup};

}

};


KernelString get_alpha_kernelstring(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const derivedparams::DerivedParams & dp){
 std::cout << "in get_alpha_kernelstring" << std::endl;

  std::string type = dp.does_beta_c_inc ? "betac_alphaab" : "alphaab";
  AlphaGenerator ag(hp, gg, dp, type);
  ag.setup();
  
  return ag.get_kernelstring();
}


}
}
