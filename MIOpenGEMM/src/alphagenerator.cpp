#include <MIOpenGEMM/alphagenerator.hpp>
#include <MIOpenGEMM/basegenerator.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <tuple>
#include <fstream>

/* TODO : reconsider half threads loading A to LDS, others B to LDS, even though will only work in unusual cirmumstances */
/* (strides same, or tiles parallel to contiguous). With normal form matrices, it should work in all situations  */
/* TODO : remove final barrier, not nec */
/* TODO : beta = 1 optimisation */
/* TODO : mad as hyper-parameter, figure out why it slows kernels down (see tensile branch) */
/* TODO : float4 */
/* TODO : volatile int id (Nugteren) to prevent unrolling */
/* TODO : play with restrict keyword */
/* TODO : add ICE > 1 with non-interwoven partitioning */





namespace MIOpenGEMM{
namespace alphagen{


class AlphaGenerator : basegen::BaseGenerator{

private:

void set_usage(){
  
  uses_a = (hp.at(nsHP::matA).vs[nsHP::WOS] == 0) ? true : false;   //this worked. propogate.
  uses_b = (hp.at(nsHP::matB).vs[nsHP::WOS] == 0) ? true : false;
  uses_c = true;
  uses_workspace = (hp.at(nsHP::matA).vs[nsHP::WOS] + hp.at(nsHP::matB).vs[nsHP::WOS]) == 0  ? false : true;
  uses_alpha  = true;
  uses_beta = dp.main_does_beta_c_inc;

}


public:
  AlphaGenerator(const hyperparams::HyperParams & hp_, const Geometry & gg_, const derivedparams::DerivedParams & dp_, std::string & type_):
  basegen::BaseGenerator(hp_, gg_, dp_, type_)  
  {
  }

virtual void setup() final override{  

  set_usage();

}


private:
  
void append_group_allocation_string(std::stringstream & ss){
  if (hp.at(nsHP::matC).vs[nsHP::GAL] == nsGAL::bycol){
    ss << 
R"(
/* GROUP_ALLOCATION = 1 :  allocation is done column-by-column */
const unsigned group_id_a = group_id_xy % N_GROUPS_A;
const unsigned group_id_b = group_id_xy / N_GROUPS_A;
)";
  }
  
  else if (hp.at(nsHP::matC).vs[nsHP::GAL] == nsGAL::byrow){
    ss << 
R"(
/* GROUP_ALLOCATION = 2 :  allocation is done row-by-row */
unsigned group_id_b = group_id_xy % N_GROUPS_B;
unsigned group_id_a = group_id_xy / N_GROUPS_B;
)";
  }
  
  else if (hp.at(nsHP::matC).vs[nsHP::GAL] == nsGAL::sucol){
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
unsigned group_id_b;
unsigned group_id_a;
unsigned wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*N_GROUPS_A);
)";


    std::string full_sucol_string = R"(
group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
group_id_a = (group_id_xy / SUPER_COLUMN_WIDTH) % N_GROUPS_A;
)";

    std::string partial_sucol_string = R"(
group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % LAST_SUPER_COLUMN_WIDTH;
group_id_a = (group_id_xy  - (N_GROUPS_B - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A) / LAST_SUPER_COLUMN_WIDTH;
)";

  
    /* super column width perfectly fits across B */
    if (dp.ga3_last_super_column_width == 0){
      ss << full_sucol_string;
    }
    
    else {
      
      /* there is just one column */    
      if (dp.ga3_last_super_column_width == dp.at(nsHP::matB).n_groups){
        ss << partial_sucol_string;
      }
      
      else{
        ss << "\n" << "if (group_id_xy < (N_GROUPS_B - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A){";
        ss << full_sucol_string << "}\n";

        ss << "else{";
        ss << partial_sucol_string << "}\n";      
      }
    }
  }

  else{
    std::stringstream err_ss;
    err_ss << "Invalid group_allocation parameter : " << hp.at(nsHP::matC).vs[nsHP::GAL] << ". It should be one of 1/2/3.";
    throw miog_error(err_ss.str());
  }
}

void append_super_column_width_defn(std::stringstream & ss){

  if (hp.at(nsHP::matC).vs[nsHP::GAL] == 3){
  
    ss <<  "\n\n" << "/* This variable defines the width of super-columns (we have GROUP_ALLOCATION 3). It is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) */\n" << "#define SUPER_COLUMN_WIDTH " << dp.ga3_super_column_width;   
    ss << "\n/* LAST_SUPER_COLUMN_WIDTH : N_GROUPS_B % SUPER_COLUMN_WIDTH  */";
    ss << "\n#define LAST_SUPER_COLUMN_WIDTH " << dp.ga3_last_super_column_width;    
  }
}

void append_split_on_k_vardecl_write_string(std::stringstream & ss){
  if (dp.main_split_on_k != 0){
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


void append_load_for_perp(char X, std::stringstream & ss){
  
  
  nsHP::eMat emat_x = hp.get_eMat_from_char(X);


  std::string bound_string = hp.at(emat_x).vs[nsHP::LIW] == 0 ? std::string("MICRO_") + X + "_TILE_PERP_UNROLL" : std::string("MACRO_TILE_LENGTH_") + X;
  std::string increment_string = hp.at(emat_x).vs[nsHP::LIW] == 0 ? "++mu_perp_i" : std::string("mu_perp_i += MACRO_TILE_LENGTH_") + X + "/MICRO_" + X + "_TILE_PERP_UNROLL";
  append_loop_var_bound_incr(ss, "mu_perp_i", bound_string, increment_string);
}

void append_load_for_pll(char X, std::stringstream & ss){

  nsHP::eMat emat_x = hp.get_eMat_from_char(X);


  std::string bound_string = hp.at(emat_x).vs[nsHP::LIW] == 0 ? std::string("MICRO_") + X + "_TILE_PLL_UNROLL" : "UNROLL";
  std::string increment_string = hp.at(emat_x).vs[nsHP::LIW] == 0 ? "++mu_pll_i" : std::string("mu_pll_i += UNROLL/MICRO_") + X + "_TILE_PLL_UNROLL";
  append_loop_var_bound_incr(ss, "mu_pll_i", bound_string, increment_string);
}



void append_final_write_element(std::stringstream & ss, unsigned atomic_increment, unsigned with_beta_scaling, unsigned with_alpha_increment){
  
  /* a good place to break kernel to check error checking. make this 1.11101242345 for example */
  std::string alpha_scaled =  "alpha*rC[row/C_INTERWEAVE_STRIDE_A][col/C_INTERWEAVE_STRIDE_B]";
  
  ss << "\nindex = STRIDE_PLL_M_C*(write_start_a + row) + STRIDE_PLL_N_C*(write_start_b + col);\n";
  /* beta string */
  ss << (with_beta_scaling == 0 ? "" : "c[index] *= beta;\n");
  if (with_alpha_increment != 0){
    ss << "\n";
    if (atomic_increment == 0){
      ss << "c[index] += " <<  alpha_scaled << + ";\n"; 
    }
    
    else{
      ss  
      << "ptr_to_c_elm = c + index;\n" 
      << "do {\n"
      << "previous_value = *ptr_to_c_elm;\n" 
      << "prevVal = as_" << dp.infa << "(previous_value);\n"
      << "newVal = as_" << dp.infa << "(" << alpha_scaled << " + previous_value);\n"
      << "} while (" << dp.fati << "(( __global " << dp.infa << "*)(ptr_to_c_elm), prevVal, newVal) != prevVal);";        
    }
  }
}

void append_for_loops_for_c_write_open(std::stringstream & ss){
  
  ss << "\n/* loops for writing to c */\n";
  
  ss << dp.pragma_unroll_string;
  append_loop_var_bound_incr(ss, "row", 
  hp.at(nsHP::matA).vs[nsHP::MIW] == 0 ? "MICRO_TILE_LENGTH_A" : "MACRO_TILE_LENGTH_A", 
  hp.at(nsHP::matA).vs[nsHP::MIW] == 0 ? "++row" : "row += N_MICRO_IN_MACRO_A");
  ss << " {\n";
  
  ss  << dp.pragma_unroll_string;
  append_loop_var_bound_incr(ss, "col",
  hp.at(nsHP::matB).vs[nsHP::MIW] == 0 ? "MICRO_TILE_LENGTH_B" : "MACRO_TILE_LENGTH_B", 
  hp.at(nsHP::matB).vs[nsHP::MIW] == 0 ? "++col" : "col += N_MICRO_IN_MACRO_B");
  ss << " {\n";
}
  
void append_for_loops_for_c_write_close(std::stringstream & ss){
  ss << "\n}\n}\n";
}

void append_check_wrapped_if_clause_open(std::stringstream & ss){
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
  if (dp.main_split_on_k == 0){
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


/* simple for loops. Could consider unrolling like Cobalt, but for the moment I use the optional pragma unroll */
void append_load_ab_into_LDS_string(std::stringstream & ss, unsigned final_unroll, unsigned special_first_unroll){

  append_load_into_LDS_string('a', ss, final_unroll, special_first_unroll);
  append_load_into_LDS_string('b', ss, final_unroll, special_first_unroll);  


  ss << 
R"(
/* make sure all loads from LDS memory have completed */
barrier(CLK_LOCAL_MEM_FENCE); )";

}



/* simple for loops. Could consider unrolling like Cobalt, but for the moment I use the optional pragma unroll */
void append_load_into_LDS_string(char x, std::stringstream & ss, unsigned final_unroll, unsigned special_first_unroll){
  
  
  char X = (x == 'a') ? 'A' : 'B';
  
  std::string n_jumps_string = dp.main_split_on_k == 0 ? "UNROLL" : "G_UNROLL";
  
  
  if (final_unroll != 0 && special_first_unroll != 0){
    throw miog_error("From get_load_ab_into_LDS_string > It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out");
  }
  
  std::stringstream ss_value_to_get;
  std::stringstream ss_comment;
  
  
  if (final_unroll == 1 || special_first_unroll == 1){
    std::string condition = final_unroll == 1 ? " < k_remaining " : " >= unroll_offset" ;
    std::string special_comment = final_unroll == 1 ? "(ignoring tail)" : "(ignoring prepend)";
    ss_value_to_get 
    << "(" << x << "_offset_pll_unroll + mu_pll_i) " << condition << " ? "  << x << "[mu_pll_i*STRIDE_PLL_K_" << X << " + mu_perp_i*STRIDE_PERP_K_" << X << "] : 0;";
    ss_comment
    <<  "/* load final bit of data from " << x << " into LDS, less than a full unroll " << special_comment << " */";
  }
  
  
  else{
    ss_value_to_get << x << "[mu_pll_i*" << "STRIDE_PLL_K_" << X  << " + mu_perp_i*" <<  "STRIDE_PERP_K_" << X <<  "];";
    ss_comment <<  "/* load data from " << x << " into LDS */";
  }
  
  ss << "\n" << ss_comment.str() << "\n"
  << dp.pragma_unroll_string;
  append_load_for_perp(X, ss);
  ss << " {\n" 
  << dp.pragma_unroll_string;
  append_load_for_pll(X, ss);
  ss << " {\n"
  << "local" << X << "[MACRO_TILE_LENGTH_" << X << "_AND_PAD*(" << x << "_offset_pll_unroll + mu_pll_i) + (" << x << "_offset_perp_unroll + mu_perp_i)] = \n" 
  << ss_value_to_get.str() << "\n" 
  <<  "}\n" 
  <<  "}\n";

  
  
  if (final_unroll == 0) ss << x << " += " << "STRIDE_PLL_K_" << X << "*" << n_jumps_string << ";\n"; 
  
  ss << "\n";
}

    

std::string get_c_work_item_next(char X){
  
  nsHP::eMat emat_x = hp.get_eMat_from_char(X);
  
  return (hp.at(emat_x).vs[nsHP::MIW] != 0) ? "1" : (std::string("MICRO_TILE_LENGTH_") + X);
}



/* We previously had a variable unroll_the_math_section = False. */
/* Experiments with unroll_the_math_section suggest that it's a bad idea. */
void append_math_section(std::stringstream & ss, unsigned use_k_remaining){
  
  std::string number_of_unrolls = use_k_remaining == 0 ? "UNROLL" : "k_remaining";
  ss << "\nfor (unsigned u = 0; u < " << number_of_unrolls << "; ++u){\n";  
  append_load_to_register_string('a', ss);
  append_load_to_register_string('b', ss);
  ss << "\n";
  append_compute_string(ss);  
  
  ss << "}\n";
}

void append_relocate_load_math_string(std::stringstream & ss, unsigned final_unroll, unsigned special_first_unroll){
  if (final_unroll != 0 && special_first_unroll != 0){
    throw miog_error("From get_relocate_load_math_string : It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out");
  }

  append_load_ab_into_LDS_string(ss, final_unroll, special_first_unroll);
  
  ss << "\n";
  for (char X : {'A', 'B'}){
    char x = (X == 'A') ? 'a' : 'b';
    ss << "\n" << "l" << X << " = local" << X << " + micro_id_" << x << "*"  << get_c_work_item_next(X) << ";";
  }
  
  ss << "\n";
  
  append_math_section(ss, final_unroll);
  ss << 
R"(
/* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
barrier(CLK_LOCAL_MEM_FENCE); )";
}
  
void append_final_unroll_string(std::stringstream & ss){
  
  if (dp.main_split_on_k == 0){
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
  if (hp.at(nsHP::matC).vs[nsHP::UFO] != 0){
    ss << "\n\n/* This is where the first unroll will be performed. Identical to what is in the main while, but with zero buffering.  */";
    if (dp.main_split_on_k == 0){
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
  << "for (unsigned row = 0; row < MICRO_TILE_LENGTH_A; ++row){\n"
  << dp.pragma_unroll_string 
  << "for (unsigned col = 0; col < MICRO_TILE_LENGTH_B; ++col){\n" 
  << "rC[row][col] += rA[row]*rB[col];   \n}\n}\n";
} 


void append_load_to_register_string(char x, std::stringstream & ss){
  char X = (x == 'a') ? 'A' : 'B';
  ss << "\n" << dp.pragma_unroll_string;
  ss << "for (unsigned i = 0; i < MICRO_TILE_LENGTH_" << X << "; ++i){\n";
  ss << "r" << X << "[i] = l" << X << "[" << "i*" << "C_INTERWEAVE_STRIDE_" << X << "];\n}\n";
  ss << "l" << X << " += MACRO_TILE_LENGTH_" << X << "_AND_PAD;\n"; 
  
}


void append_group_allocation_defn_string(std::stringstream & ss){
  ss << "#define GROUP_ALLOCATION " << hp.at(nsHP::matC).vs[nsHP::GAL] << "\n";
  if (hp.at(nsHP::matC).vs[nsHP::GAL] == 3){
    ss << "/* this variable is declared because we have GROUP_ALLOCATION type 3. */\n";
    ss << "/* It should define how many workgroups we expect to have active simulantaneuosly. */\n";
    ss << "#define N_TARGET_ACTIVE_WORKGROUPS " << hp.at(nsHP::matC).vs[nsHP::NAW] << "\n";
  }
}



void append_final_write_all(std::stringstream & ss){
  
  
  
  if (dp.main_use_edge_trick == 0){
    ss << "\n";
    append_final_write_loops_no_check(ss);
  }
  
  else{
        
    std::string cond_a("");
    if (dp.at(nsHP::matA).preshift_final_tile != dp.at(nsHP::matA).macro_tile_length){
      cond_a = "(group_id_a != N_GROUPS_A - 1)";
    }


    std::string cond_b("");
    if (dp.at(nsHP::matB).preshift_final_tile != dp.at(nsHP::matB).macro_tile_length){
      cond_b = "(group_id_b != N_GROUPS_B - 1)";
    }

    
    if (cond_a == "" && cond_b == ""){
      append_final_write_loops_no_check(ss);
    }
    
    else{
      
      if (dp.main_use_edge_trick == 0){
        throw miog_error("in alphagenerator, dp.main_use_edge_trick == 0. however, non-perfectly tilable");
      }
      
      ss << "\n/* the case where this is not an edge tile : will write to all cells */ \n";
      if (cond_a != "" && cond_b != ""){
        ss << "if (" << cond_a << " && " << cond_b << "){ \n";
      }
      else if (cond_a != ""){
        ss << "if  " << cond_a << "{ \n";
      }
      else{
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
    


void append_split_on_k_defns_string(std::stringstream & ss){
  if (dp.main_split_on_k != 0){
    ss << 
R"(/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL )" << hp.at(nsHP::matC).vs[nsHP::ICE]*hp.at(nsHP::matC).vs[nsHP::UNR] << " // N_WORK_ITEMS_PER_C_ELM*UNROLL";
  }
}

void append_group_id_defns(std::stringstream & ss){
  if (dp.main_split_on_k == 0){
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

void append_stride_c_defn(std::stringstream & ss){
  
  unsigned transposed_xor_is_col_major = (gg.tX[nsHP::matC] + gg.isColMajor) % 2;
  ss << "#define STRIDE_PLL_M_C " << (transposed_xor_is_col_major == 1 ? 1 : gg.ldX[nsHP::matC]) << "\n";
  ss << "#define STRIDE_PLL_N_C " << (transposed_xor_is_col_major == 0 ? 1 : gg.ldX[nsHP::matC]) << "\n";
}


void append_n_unrolls_remaining_string(std::stringstream & ss){
  std::string k_effective_mod_G_UNROLL = dp.effective_k_varies_string + " % G_UNROLL";
  std::string k_effective_div_G_UNROLL = dp.effective_k_varies_string + " / G_UNROLL";
  std::string k_effective_div_UNROLL = dp.effective_k_varies_string + " / UNROLL";
  
  if (dp.main_split_on_k == 0){
    ss << "\nint n_unrolls_remaining = " << k_effective_div_UNROLL << ";";
  }
  
  else{    
    ss << "\n/* a certain number of work groups process one more unroll. Note that with UFO = 1, this depends on column */";
    ss << "\nconst int n_work_groups_with_1_more = (" << k_effective_mod_G_UNROLL << ") / UNROLL; \n";
    ss << "\n/* branching between work groups : some wgs have 1 more unroll to process. */\n";
    ss << "int n_unrolls_remaining = (" << k_effective_div_G_UNROLL;
    ss << ") +  (group_id_z < n_work_groups_with_1_more);";
  }
}


void append_c_offset_string(std::stringstream & ss){

  ss << R"(

/* In OpenCL, host code does not have access to raw data pointers. */
/* Host code works with cl_mem objects, which encapsulate and hide raw points. */
/* For this reason, host code CANNOT simply increment pointers to data, */
/* as one can do with pointers for CPU gemm, or cublas gemm.*/

c += c_offset;
)";
}


void append_id_string_nonsym(std::stringstream & ss){
  ss << "const unsigned local_id = get_local_id(0);\n";
  append_group_id_defns(ss);

  ss << R"(
/* Define which part of the C macro-tile this thread will process (% / or / % ? doesn't seem to make much difference) */
const unsigned micro_id_a = local_id % N_MICRO_IN_MACRO_A;
const unsigned micro_id_b = local_id / N_MICRO_IN_MACRO_A;
)";


  append_group_allocation_string(ss);
  
  if (hp.at(nsHP::matC).vs[nsHP::UFO] != 0){
    ss <<
R"(
/* this additional offset of a and b appears because UNROLL_FOR_OFFSET is 1 */
unsigned unroll_offset = (13*group_id_a + 7*group_id_b)%UNROLL;
unsigned k_plus_offset = __K__ + unroll_offset;
)";
  }
}



void append_id_string_sym(std::stringstream & ss, char x){

  /* set upper - lower case versions */
  char X = 'Z';
  X = (x == 'a') ? 'A' : ((x == 'b') ? 'B' : x);
  x = (x == 'B') ? 'b' : ((x == 'A') ? 'a' : x);

  nsHP::eMat emat_x = hp.get_eMat_from_char(x);


  ss << "\n";  
  
  if (X == 'A') ss << "/* LDS memory */\n";
  ss << "__local TFLOAT local" << X << "[N_ELEMENTS_IN_PADDED_" << X << "_UNROLL];\n";
  if (X == 'A') ss << "/* jumping pointer to locate the LDS to load into register memory */\n";
  ss << "__local const TFLOAT * l" << X << ";\n";
  if (X == 'A') ss << "/* register memory */ \n";
  ss << "TFLOAT r" << X << "[MICRO_TILE_LENGTH_" << X << "];\n";
  if (X == 'A') ss << "/* Define which part of the C macro-tile this thread will process (% / or / % ? doesn't seem to make much difference) */\n";
  ss << "unsigned write_macro_tile_start_" << x << " = group_id_" << x << "*MACRO_TILE_LENGTH_" << X << "; \n";
  if (dp.main_use_edge_trick != 0){
    if (X == 'A') ss << "/* tile on edge : pulling it in so no C overflow */\n";
    ss << "if (group_id_" << x << " == N_GROUPS_" << X << " - 1){\n";
    ss << "write_macro_tile_start_" << x << " -= (MACRO_TILE_LENGTH_" << X << " - PRESHIFT_FINAL_TILE_" << X << ");\n";
    ss << "}\n";
  }
  ss << "const unsigned write_start_" << x << " = write_macro_tile_start_" << x << " + micro_id_" << x << "*" << get_c_work_item_next(X) << ";\n";
 
  ss << "\n\n\n"; 
  
  
  if (hp.at(emat_x).vs[nsHP::WOS] == 1 || hp.at(emat_x).vs[nsHP::WOS] == 2){    
    if (X == 'A') ss << "/* from workspace */\n";
    ss << "const TFLOAT * restrict " << x << " = w + w_offset + GLOBAL_OFFSET_" << X << ";\n";
  }

    
  else{
    ss << x << " += " << x << "_offset;\n";
  }

  if (X == 'A') ss << "/* Define what of A this thread will load from unroll tile in global to LDS (% / or / % ? looks like no difference ) */\n";
  ss << "const unsigned pll_unroll_" << x << "_load_id = local_id % N_MICRO_" << X << "_TILES_PLL_UNROLL;\n";
  ss << "const unsigned perp_unroll_" << x << "_load_id = local_id / N_MICRO_" << X << "_TILES_PLL_UNROLL;\n";


  if (X == 'A') ss << "/* Define which part of A this thread will read from process (% / or / % ? doesn't seem to make much difference) */\n";
  ss << "unsigned read_macro_tile_start_" << x << " = group_id_" << x << "*MACRO_TILE_LENGTH_" << X << "; \n";  
  if (dp.main_use_edge_trick != 0 && hp.at(emat_x).vs[nsHP::WOS] != 2 ) {
    if (X == 'A') ss << "/* tile on edge and A is not normal form: pulling in read zone so no C overflow */\n";
    ss << "if (group_id_" << x << " == N_GROUPS_" << X << " - 1){\n";
    ss << "read_macro_tile_start_" << x << " -= (MACRO_TILE_LENGTH_" << X << " - PRESHIFT_FINAL_TILE_" << X << ");\n";
    ss << "}\n";
  }
  
   
  if (X == 'A') ss << "/* move to corner of the region required by the macro tile */\n";
  ss << x << " += read_macro_tile_start_" << x << "*MACRO_STRIDE_PERP_K_" << X << ";\n";
  
 
  if (dp.main_split_on_k != 0){
    if (X == 'A') ss <<  R"(/* a points to top left of region required, but this work group  */
/* might not process the whole of a. So turn 90 and move to the start for this wg */
)";
    ss << x << " += UNROLL*group_id_z*STRIDE_PLL_K_" << X << ";\n";
  }
  
  if (hp.at(nsHP::matC).vs[nsHP::UFO] != 0){
    if (X == 'A') ss << "/* UFO != 0, so offsetting the unroll */\n";
    ss << x << " -= unroll_offset*STRIDE_PLL_K_" << X << ";\n";
  }
  
  std::string str_n_pll(""); 
  std::string str_n_perp(""); 
  if (hp.at(emat_x).vs[nsHP::LIW] == 0){
    str_n_pll = std::string("MICRO_") + X + "_TILE_PLL_UNROLL *";
    str_n_perp = std::string("MICRO_") + X + "_TILE_PERP_UNROLL *";
  }
  if (X == 'A') ss << "/* make the micro adjustments (A) for the thread, getting ready to load */\n";
  ss << "const unsigned " << x << "_offset_pll_unroll = " << str_n_pll << " pll_unroll_" << x <<"_load_id;\n";
  ss << "const unsigned " << x << "_offset_perp_unroll = " << str_n_perp <<  " perp_unroll_" << x <<"_load_id;\n";
  ss << x << " += " << "STRIDE_PLL_K_" << X << " * " << x << "_offset_pll_unroll;\n";
  ss << x << " += " << "STRIDE_PERP_K_" << X << " * " << x << "_offset_perp_unroll;\n";


  ss << "\n";
  
}


void append_transpose_note(std::stringstream & ss){
  ss << R"(
/* A note on how transposes isColMajor  effect the kernel generated: very little. */
/* just via STRIDE_PLL_K_{A,B}, STRIDE_PERP_K_{A,B}, STRIDE_PLL_M_C, STRIDE_PERP_M_C */
)";
}




void add_predefine_chiral(char x, std::stringstream & ss){
  
  nsHP::eMat emat_x = hp.get_eMat_from_char(x);
  
  auto defcom = [x, &ss](std::string && comment){
    if (x == 'A') ss << "/*" << " " << comment << " : */\n";
  };    

  
  bool withcomments = x == 'A';
  bool with_x_in_name = true;
  append_unroll_block_geometry(x, ss, withcomments, with_x_in_name);

    
  
  append_stride_definitions(x, ss, hp.at(emat_x).vs[nsHP::WOS], withcomments, "", with_x_in_name);    
    
    
    
  
  if (x == 'A') ss << "/* micro tiles define the pattern of C that individual threads process */\n";
  ss << "#define MICRO_TILE_LENGTH_" << x << " " << hp.at(emat_x).vs[nsHP::MIC] << "\n";

  if (x == 'A') ss << "/* the amount of padding of " << x << " in LDS (local) memory, to avoid bank comflicts */\n";
  ss << "#define PAD_LDS_" << x << "  " << hp.at(emat_x).vs[nsHP::PAD] << "\n";
  if (x == 'A') ss << "/* whether loading of " << x << " from global should try to be long in direction of unroll (1) or perpendicular to it (0) */\n";
  ss << "#define WORK_ITEM_LOAD_" << x << "_PLL_TO_UNROLL " << hp.at(emat_x).vs[nsHP::PLU] << "\n"; 
  defcom("MACRO_TILE_LENGTH_A + PAD_LDS_A");  
  ss << "#define MACRO_TILE_LENGTH_" << x << "_AND_PAD "<< dp.at(emat_x).main_macro_tile_length_and_pad << "\n";

  defcom("MACRO_TILE_LENGTH_A_AND_PAD * UNROLL");
  ss << "#define N_ELEMENTS_IN_PADDED_" << x << "_UNROLL "<< dp.at(emat_x).main_n_elements_in_padded_unroll <<"\n";
  defcom("N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP");
  ss << "#define N_ELEMENTS_OF_" << x << "_TO_LOAD_PER_WORKITEM "<< dp.at(emat_x).main_n_elements_to_load_per_workitem <<"\n";
  defcom("MACRO_TILE_LENGTH_A / MICRO_TILE_LENGTH_A");
  ss << "#define N_MICRO_IN_MACRO_" << x << "  " << dp.at(emat_x).main_n_micro_in_macro << "\n"; 
  defcom("MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM");  
  ss << "#define MICRO_" << x << "_TILE_PLL_UNROLL " << dp.at(emat_x).main_micro_tile_pll_unroll << " \n";
  ss << "#define MICRO_" << x << "_TILE_PERP_UNROLL " << dp.at(emat_x).main_micro_tile_perp_unroll << "\n";
  
  defcom("MACRO_TILE_LENGTH_A / MICRO_A_TILE_PLL_UNROLL");  
  ss << "#define N_MICRO_" << x << "_TILES_PLL_UNROLL " << dp.at(emat_x).main_n_micro_tiles_pll_unroll << " \n";
  if (x == 'A') ss << "/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if the load tiles are truly contiguous tiles (0) */\n";
  ss << "#define LOAD_TO_LDS_INTERWOVEN_" << x << " " << hp.at(emat_x).vs[nsHP::LIW] << "\n";
  if (x == 'A') ss << "/* Whether micro tile being processed by a compute item is interwoven with other micro tiles (ala Cobalt, (1)) or if the micro tiles are contiguous in C */\n";
  ss << "#define C_MICRO_TILES_INTERWOVEN_" << x << " " << hp.at(emat_x).vs[nsHP::MIW] << "\n";


  if (x == 'A') ss << "/* depending on whether loads to c are interwoven, set as MIW == 0 ? 1 : N_MICRO_IN_MACRO_A */\n";
  ss << "#define C_INTERWEAVE_STRIDE_" << x << " " << dp.at(emat_x).main_c_interweave_stride << "\n";
  
  if (hp.at(emat_x).vs[nsHP::WOS] != 0){
    if (x == 'A') ss << "/* global memory offset, depends on type of copy of both a,b */\n";
    ss << "#define GLOBAL_OFFSET_" << x << " " << dp.at(emat_x).cw_global_offset;
  }

  ss << "\n";
}


  
public:

/* the "main" kernel */
KernelString get_kernelstring(){
  
  std::stringstream ss;
  ss << get_time_string();
  ss <<  "\n\n"; 
  ss << "/* this kernel was generated for starting geometry : */\n";
  ss << "/* " << gg.get_string() << "*/\n";  
  ss << "#define __K__ " << gg.k << "\n";
  ss << "#define TFLOAT  "  << dp.t_float << "\n";  
  ss << "#define DOES_BETA_C_INC " << dp.main_does_beta_c_inc << "\n";
  ss << "#define DOES_ALPHA_A_B_INC 1" << "\n";
  
  append_transpose_note(ss);
  
  ss << 
R"(

)";

  for (auto x : {'A', 'B'}){
    ss << "\n/* ********************************** specific to " << x << " *************************************** */";
    add_predefine_chiral(x, ss);
  }

  ss << "\n/* ********************************** common to A and B *************************************** */";
  ss << "\n/* whether or not to shimmy the starting k, in an attempt to avoid cache line overuse for cases where lda/ldb are powers of 2 */\n";
  ss << "/* if 0, no shimmying. if 1, instead of starting at k = 0 workgroups start at some negative offset dependent on work group id */\n";
  ss << "/* in the same way as the final unroll populates LDS with zeros in k mod UNROLL != 0, the initial negative indices here populate with 0 */\n";
  ss << "#define UNROLL_FOR_OFFSET " << hp.at(nsHP::matC).vs[nsHP::UFO] << "\n";
  
  ss << "/* How much a workgroup loads (global -> LDS) in the k-direction at each iteration of the outer-most loop */\n";
  ss << "#define UNROLL " << hp.at(nsHP::matC).vs[nsHP::UNR]  << "\n";
  ss << "/* whether or not this kernel uses the edge trick (SC17 submission) */\n";
  ss << "/* this precompiler defn has no direct influence on the running the kernel, implementation already done in make_kernel.py */\n";
  ss << "#define EDGETRICK " << dp.main_use_edge_trick << "\n";  
  ss << "/* the number of work items working on the same c element. if this is 1, there will be just one thread doing all k multiply-adds, */\n";
  ss << "/* otherwise if it is greater than 1, each thread will be computing ~ k / N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added at the end */ \n";
  ss << "#define N_WORK_ITEMS_PER_C_ELM " << hp.at(nsHP::matC).vs[nsHP::ICE] << "\n";
  
  ss << R"(/* define the way in which work groups are assigned to tiles */
/* 1 : column-by-column
 * 2 : row-by-row 
 * 3 : by rows within super-column  */
)";

  append_group_allocation_defn_string(ss);
  
  


  ss << "/* Whether to use the unroll pragma to encourage the compiler to unroll certain loops */\n";
  ss << "/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */\n";
  ss << "#define PRAGMA_UNROLL_FORLOOPS " << hp.at(nsHP::matC).vs[nsHP::PUN] << "\n";
  ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into registers, as compared to doing the math. */\n";
  ss << "/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */\n";
  ss << "#define N_PREFETCH_FOR_REGISTER_LOAD " << 0 << "\n";
  ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into LDS, as compared to the unroll loop */\n";
  ss << "/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */\n";
  ss << "#define N_PREFETCH_FOR_LDS_LOAD " << 0 << "\n";
  ss << "#define MACRO_TILE_AREA "<< dp.main_macro_tile_area << "\n"; // <<"  // MACRO_TILE_LENGTH_B*MACRO_TILE_LENGTH_A\n";
  ss << "#define MICRO_TILE_AREA "<< dp.main_micro_tile_area << "\n"; //" // MICRO_TILE_LENGTH_B * MICRO_TILE_LENGTH_A\n";
  ss << "#define N_WORK_ITEMS_PER_WORKGROUP  "<< dp.main_n_work_items_per_workgroup << "\n"; //" // MACRO_TILE_AREA / MICRO_TILE_AREA\n";
  ss << "/* two more parameters, which do dot have an effect the running of this kernel (used in enqueuing) */\n";
  ss << "/* the total number of work groups this kernel will use (recall m,n,k are fixed) */ \n";
  ss << "/* N_WORK_ITEMS_PER_C_ELM * ((M/MACRO_TILE_LENGTH_A) + (M%MACRO_TILE_LENGTH_A != 0)) * ((N/MACRO_TILE_LENGTH_B) + (N%MACRO_TILE_LENGTH_B != 0)) */ \n";
  ss << "#define N_WORK_GROUPS " << dp.main_n_work_groups << "\n";
  ss << "/* the global work size, ie the total mumber of work items (threads) which will run */ \n";
  ss << "/* N_WORK_GROUPS * N_WORK_ITEMS_PER_WORKGROUP */ \n";
  ss << "#define GLOBAL_WORK_SIZE " << dp.main_global_work_size << "\n";
  
  append_stride_c_defn(ss);
  append_split_on_k_defns_string(ss);
  append_super_column_width_defn(ss);
   
  ss << "\n\n\n__attribute__((reqd_work_group_size(" << dp.main_n_work_items_per_workgroup << ",1, 1)))\n";
  ss << "__kernel void ";
  ss << kernelname;
  append_parameter_list_from_usage(ss);
  
  ss << "\n{\n\n";

  append_c_offset_string(ss);

  append_id_string_nonsym(ss);
  ss << "\n\n/* *************************** A setup ************************** */";
  append_id_string_sym(ss, 'A');
  ss << "\n\n/* *************************** B setup ************************** */";
  append_id_string_sym(ss, 'B');
  
  ss << "\n\n\n";

  ss << "/* register memory for C */\n ";
  ss << "TFLOAT rC[MICRO_TILE_LENGTH_A][MICRO_TILE_LENGTH_B] = {{0.}};\n";
  

  append_n_unrolls_remaining_string(ss);
  append_first_unroll_block(ss);

  ss << "\n\nwhile (n_unrolls_remaining > 0){\n";
  append_relocate_load_math_string(ss, 0,0);
  ss << "\n--n_unrolls_remaining;\n}\n";
  
  if (dp.main_final_fractional_unroll == 1){
    ss << "\n/* *********** processing the tail *************** */\n";
    append_k_remaining_string(ss);
    append_final_unroll_string(ss);
    ss << "\n/* *********************************************** */\n\n";
  }
  
  ss << "\n\n";
  ss << "unsigned index;\n";  
  append_split_on_k_vardecl_write_string(ss);
  append_final_write_all(ss);
  ss << "\n}\n";
  
  return { {uses_a, uses_b, uses_c, uses_workspace, uses_alpha, uses_beta } , ss.str(), kernelname, dp.main_global_work_size, dp.main_n_work_items_per_workgroup};

}

};


KernelString get_alpha_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp){


  std::string type = dp.main_does_beta_c_inc ? "betac_alphaab" : "alphaab";
  AlphaGenerator ag(hp, gg, dp, type);
  ag.setup();
  


  return ag.get_kernelstring();
}


}
}
