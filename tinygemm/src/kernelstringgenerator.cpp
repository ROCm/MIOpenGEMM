#include "tinygemm/kernelstringgenerator.hpp"


#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <tuple>
#include <fstream>


namespace tinygemm{
namespace kerngen{



std::vector<unsigned> get_multiples(unsigned N){
  std::vector<unsigned> multiples;
  for (unsigned k = N; k > 0; --k){
    if (N%k == 0){
      multiples.push_back(k);
    }
  }
  return multiples;
}


/*given a macro tile TH x TW, 
  and given a micro tile size of tS, 
  find the tallest possible micro tile size (tH x tW)
  to fit the macro tile. Example, macro tile is 6 x 4:
  
  * * * * 
  * * * * 
  * * * * 
  * * * * 
  * * * * 
  * * * * 
  
  tS = 2 return [2, 1]
  tS = 3 return [3, 1]
  tS = 4 return [2, 2]
  tS = 5 raise an error ((TH * TH) % tS != 0)
  tS = 6 return [6, 1]
  tS = 7 raise an error ((TH * TH) % tS != 0) 
  tS = 8 return [2, 4]
  tS = 9 raise an error ((TH * TH) % tS != 0)
  tS = 10 raise an error ((TH * TH) % tS != 0)
  tS = 11 raise an error ((TH * TH) % tS != 0)
  tS = 12 return [6, 2]
  tS = 13 .. 23 raise an error ((TH * TH) % tS != 0)
  tS = 24 return [6, 4] */



class KernelString{

  private:
    /* to be set in constructor based on passed in parameters */
    

    /* */
    std::string strided_i_vertical;
    /* */
    std::string strided_i_horizontal; 
    /*the int type for atomics */
    std::string infa;
    /* the function to use for atomic ints */
    std::string fati;
    /* */
    std::string effective_k; 
    /* */
    std::string alpha_scaled;
    /* pragma unroll string ("#pragma unroll\n" or "") */
    std::string pragma_unroll_string;
    /* currently one of "float" and "double", set from float_size */
    std::string t_float;
    unsigned split_on_k;
    unsigned does_alpha_c_inc;
    unsigned macro_tile_area;
    unsigned micro_tile_area;
    unsigned n_workitems_per_workgroup;
    unsigned macro_tile_height_and_pad;
    unsigned macro_tile_width_and_pad;
    unsigned n_elements_in_a_unroll;
    unsigned n_elements_in_b_unroll;
    unsigned n_elements_in_padded_a_unroll;
    unsigned n_elements_in_padded_b_unroll;
    unsigned n_elements_of_a_to_load_per_workitem;
    unsigned n_elements_of_b_to_load_per_workitem;
    unsigned n_micro_tiles_vertically;
    unsigned n_micro_tiles_horizontally;
    unsigned micro_a_tile_pll_unroll; 
    unsigned micro_a_tile_perp_unroll; 
    unsigned n_micro_a_tiles_pll_unroll;  
    unsigned micro_b_tile_pll_unroll;
    unsigned micro_b_tile_perp_unroll;
    unsigned n_micro_b_tiles_pll_unroll;
    
    /* set here as we do not want the user to choose (although will not break kernel) */
    unsigned use_edge_trick = 1;

  public:

    std::string kernelname;
    
    unsigned float_size;
    // tA  
    unsigned a_transposed;
    // tB
    unsigned b_transposed; 
    // tC
    unsigned c_transposed;
    // CM
    unsigned is_col_major; 
    // x
    unsigned micro_tile_width; 
    // y
    unsigned micro_tile_height;
    // X
    unsigned macro_tile_width; 
    // Y
    unsigned macro_tile_height; 
    // U
    unsigned unroll;
    // P
    unsigned pad;
    // GA
    unsigned group_allocation;
    // APLU
    unsigned work_item_load_a_pll_to_unroll;
    // BPLU
    unsigned work_item_load_b_pll_to_unroll; 
    // PU
    unsigned unroll_pragma;
    // LIW
    unsigned load_to_lds_interwoven;
    // MIW
    unsigned c_micro_tiles_interwoven; 
    // ICE
    unsigned n_work_items_per_c_elm;
    // UFO
    unsigned unroll_for_offset;
    // haven't optimised over this
    unsigned n_target_active_workgroups;  
  
  KernelString(
  std::string kernelname = "",
  
  /* geometry parameters */
  unsigned float_size = 32,
  unsigned a_transposed = 1,
  unsigned b_transposed = 0,
  unsigned c_transposed = 0,
  unsigned is_col_major = 1,
  /* to add m,n,k, lda, ldb, ldc */
  
  
  /* hyper parameters */
  unsigned micro_tile_width = 3, 
  unsigned micro_tile_height = 2, 
  unsigned macro_tile_width = 48, 
  unsigned macro_tile_height = 32, 
  unsigned unroll = 16, 
  unsigned pad = 1, 
  unsigned group_allocation = 2, 
  unsigned work_item_load_a_pll_to_unroll = 0, 
  unsigned work_item_load_b_pll_to_unroll = 1, 
  unsigned unroll_pragma = 1, 
  unsigned load_to_lds_interwoven = 0, 
  unsigned c_micro_tiles_interwoven = 1, 
  unsigned n_work_items_per_c_elm = 3,
  unsigned unroll_for_offset = 1,
  unsigned n_target_active_workgroups = 64):
  
  kernelname(kernelname),
  float_size(float_size),
  a_transposed(a_transposed), 
  b_transposed(b_transposed), 
  c_transposed(c_transposed), 
  is_col_major(is_col_major), 
  micro_tile_width(micro_tile_width), 
  micro_tile_height(micro_tile_height), 
  macro_tile_width(macro_tile_width), 
  macro_tile_height(macro_tile_height), 
  unroll(unroll), 
  pad(pad), 
  group_allocation(group_allocation), 
  work_item_load_a_pll_to_unroll(work_item_load_a_pll_to_unroll), 
  work_item_load_b_pll_to_unroll(work_item_load_b_pll_to_unroll), 
  unroll_pragma(unroll_pragma), 
  load_to_lds_interwoven(load_to_lds_interwoven), 
  c_micro_tiles_interwoven(c_micro_tiles_interwoven), 
  //use_edge_trick(use_edge_trick), 
  n_work_items_per_c_elm(n_work_items_per_c_elm),
  unroll_for_offset(unroll_for_offset), 
  n_target_active_workgroups(n_target_active_workgroups){

    
    
    split_on_k = n_work_items_per_c_elm == 1 ? 0 : 1;
    does_alpha_c_inc = split_on_k == 1 ? 0 : 1; 
    strided_i_vertical = c_micro_tiles_interwoven == 0 ? "i" : "i*N_MICRO_TILES_VERTICALLY";
    strided_i_horizontal = c_micro_tiles_interwoven == 0 ? "i" : "i*N_MICRO_TILES_HORIZONTALLY";
    
    if (n_work_items_per_c_elm == 1){
      infa = "n_work_items_per_c_elm is 1, should not be using atomics";
      fati = "n_work_items_per_c_elm is 1, should not be using atomics";
    }
    else{
      infa = float_size == 32 ? "uint" : "ulong";
      fati = float_size == 32 ? "atomic_cmpxchg" : "atom_cmpxchg";
    }
    
    pragma_unroll_string = unroll_pragma == 1 ?  "#pragma unroll\n" : "" ;
    
    effective_k = unroll_for_offset == 0 ? "k" : "k_plus_offset";    
    alpha_scaled = c_micro_tiles_interwoven == 0 ? "alpha*rC[row][col]" : "alpha*rC[row/N_MICRO_TILES_VERTICALLY][col/N_MICRO_TILES_HORIZONTALLY]";
    t_float = float_size == 32 ? "float" : "double";
    
    macro_tile_area = macro_tile_width * macro_tile_height;
    micro_tile_area = micro_tile_width * micro_tile_height;
    n_workitems_per_workgroup = macro_tile_area / micro_tile_area;
    macro_tile_height_and_pad = macro_tile_height + pad;
    macro_tile_width_and_pad = macro_tile_width + pad;
    n_elements_in_a_unroll = macro_tile_height * unroll;
    n_elements_in_b_unroll = macro_tile_width * unroll;
    n_elements_in_padded_a_unroll = macro_tile_height_and_pad * unroll;
    n_elements_in_padded_b_unroll = macro_tile_width_and_pad * unroll;
    n_elements_of_a_to_load_per_workitem = n_elements_in_a_unroll / n_workitems_per_workgroup;
    n_elements_of_b_to_load_per_workitem = n_elements_in_b_unroll / n_workitems_per_workgroup;
    n_micro_tiles_vertically = macro_tile_height / micro_tile_height;
    n_micro_tiles_horizontally = macro_tile_width / micro_tile_width;
    
    
    test_parameters_prestringbuild();
    
      

   
     
    
    
  }


  std::string 
  set_tile_dimensions(unsigned & tH, unsigned & tW, unsigned TH, unsigned TW, unsigned tS){

    std::string set_ds("");
  
    std::stringstream err_ss;
    err_ss << get_string() << "\n" << "TH : " << TH << " TW : " << TW << " tS : " << tS;
  
    if ((TH*TW) % tS  != 0){
      set_ds += "Areas of micro and macro tiles are incompatible : ";
      set_ds += err_ss.str();
      return set_ds;
    }
    
    for (auto & multiple_of_TH : get_multiples(TH)){
      if ((tS % multiple_of_TH == 0) && ((tS / multiple_of_TH) <= TW)){
        tH = multiple_of_TH;
        tW = tS / tH;
        break;
      }
    }
    
    if (tH == 0){
      set_ds += "Impossible tiling problem in get_tile_dimensions : ";
      set_ds += err_ss.str();
      return set_ds;
    }
    
    err_ss << " tH : " << tH << " tW  " << tW;
  
    if (tW  > tH){
      //throw std::runtime_error("this is a pedantic error, can remove when confirmed no prob: no `tall' tile. best `wide' one " + err_ss.str());
    }
    
    if (TW % tW != 0 || TH % TH != 0 || tW*tH != tS){
      throw std::runtime_error("This is strange : the found micro tile size is not consistent with the macro tile : "  + err_ss.str());
    }
    
    return set_ds; 
  }
  
  
  std::string get_string(){
    std::stringstream ss;
    ss << "Y" << macro_tile_height <<  "_X" << macro_tile_width << "_y" << micro_tile_height << "_x" << micro_tile_width << "_U" << unroll << "_P" << pad << "_GA" << group_allocation << "_APLU" << work_item_load_a_pll_to_unroll << "_BPLU" << work_item_load_b_pll_to_unroll << "_PU" << unroll_pragma << "_LIW" << load_to_lds_interwoven << "_MIW" << c_micro_tiles_interwoven  << "_ET" << 1 << "_ICE" << n_work_items_per_c_elm << "_UFO" << "unroll_for_offset";
    
    return ss.str();
    
    
  }
  
  
  
  /* TODO : streamline this, so that all parameters which should be 0 or 1 are processed in a for loop. */
  /* TODO : gather all errors and put them here */
  void test_parameters_prestringbuild(){
    
    std::stringstream errstream;
    if (load_to_lds_interwoven != 0 && load_to_lds_interwoven != 1){
      errstream << "load_to_lds_interwoven is neither 0 nor 1 sdfaeff\n";
    }
    
    if (t_float != "float" && t_float != "double"){
      errstream << "in get_inttype_for_atomics in make_kernel, and don't recognise the t_float, " <<  t_float << "\n";
    }
    
    if (group_allocation != 1 && group_allocation != 2 && group_allocation != 3){
      errstream << "Invalid group_allocation value, it should be in [1,2,3], not " << group_allocation << "\n";
    }

    if (work_item_load_a_pll_to_unroll != 0 && work_item_load_a_pll_to_unroll != 1){
      errstream << "this is strange, not sure what to do with `work_item_load_a_pll_to_unroll = " << work_item_load_a_pll_to_unroll << " . It 'should be 0 (for false) or 1 (for true).";
    }

    if (work_item_load_b_pll_to_unroll != 0 && work_item_load_b_pll_to_unroll != 1){
      errstream << "this is strange, not sure what to do with `work_item_load_b_pll_to_unroll = " << work_item_load_b_pll_to_unroll << " . It 'should be 0 (for false) or 1 (for true).";
    }    
    
    auto errstring = errstream.str();
    if (errstring.compare("") != 0){
      throw std::runtime_error(errstring);
    }
  }
  
    
  void append_group_allocation_string(std::stringstream & ss){
    if (group_allocation == 1){
      ss << 
R"(
/* GROUP_ALLOCATION = 1 :  allocation is done column-by-column */
const unsigned group_id_vertical = group_id_xy % n_groups_vertically;
const unsigned group_id_horizontal = group_id_xy / n_groups_vertically;
)";
    }
    
    else if (group_allocation == 2){
      ss << 
R"(
/* GROUP_ALLOCATION = 2 :  allocation is done row-by-row */
unsigned group_id_horizontal = group_id_xy % n_groups_horizontally;
unsigned group_id_vertical = group_id_xy / n_groups_horizontally;
)";
    }
    
    else if (group_allocation == 3){
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
unsigned wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*n_groups_vertically);
unsigned last_super_column_width = n_groups_horizontally % SUPER_COLUMN_WIDTH;
    
if (group_id_xy < (n_groups_horizontally - last_super_column_width)*n_groups_vertically){
group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
group_id_vertical = (group_id_xy / SUPER_COLUMN_WIDTH) % n_groups_vertically;
}
else{
group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % last_super_column_width;
group_id_vertical = (group_id_xy  - (n_groups_horizontally - last_super_column_width)*n_groups_vertically) / last_super_column_width;
}
)";
  
    }
    
    else{
      std::stringstream err_ss;
      err_ss << "Invalid group_allocation parameter : " << group_allocation << ". It should be one of 1/2/3.";
      throw std::runtime_error(err_ss.str());
    }
  }

  void append_super_column_width_defn(std::stringstream & ss){

    if (group_allocation == 3){

      unsigned super_column_width;
      if (split_on_k == 1){
        super_column_width = 
        static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(n_target_active_workgroups) / static_cast<double>(n_work_items_per_c_elm))));
      }
      else if (split_on_k == 0){
        super_column_width = 
        static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(n_target_active_workgroups))));
      }
      
      else{
        throw std::runtime_error("split_on_k is neither 0 nor 1, how can this be? Logic error in append_super_column_width_defn");
      }
      
      ss <<  "\n" << "/* This variable defines the width of super-columns (we have GROUP_ALLOCATION 3). It is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) */\n" << "#define SUPER_COLUMN_WIDTH " << super_column_width;
    }
  }
  
  void append_split_on_k_vardecl_write_string(std::stringstream & ss){
    if (split_on_k != 0){
      ss << 
R"(
/* the following variables are used in implementing a basic atomic increment */
global TFLOAT * ptr_to_c_elm;  // with `restrict' is no faster
TFLOAT previous_value; )" << "\n" << infa << " newVal;\n" << infa << " prevVal;" << "\n\n";
    }
  }
  
  void append_loop_var_bound_incr(std::stringstream & ss, std::string varname, std::string bound_string, std::string increment_string){
    ss << "for (unsigned " << varname << " = 0; " << varname <<  " < " << bound_string  << "; " <<  increment_string << ")";
  }
  
  
  void append_a_load_for_perp(std::stringstream & ss){
    std::string bound_string = load_to_lds_interwoven == 0 ? "MICRO_A_TILE_PERP_UNROLL" : "MACRO_TILE_HEIGHT";
    std::string increment_string = load_to_lds_interwoven == 0 ? "++mu_perp_i" : "mu_perp_i += MACRO_TILE_HEIGHT/MICRO_A_TILE_PERP_UNROLL";
    append_loop_var_bound_incr(ss, "mu_perp_i", bound_string, increment_string);
  }
  
  void append_a_load_for_pll(std::stringstream & ss){
    std::string bound_string = load_to_lds_interwoven == 0 ? "MICRO_A_TILE_PLL_UNROLL" : "UNROLL";
    std::string increment_string = load_to_lds_interwoven == 0 ? "++mu_pll_i" : "mu_pll_i += UNROLL/MICRO_A_TILE_PLL_UNROLL";
    append_loop_var_bound_incr(ss, "mu_pll_i", bound_string, increment_string);
  }

  void append_b_load_for_perp(std::stringstream & ss){
    std::string bound_string = load_to_lds_interwoven == 0 ? "MICRO_B_TILE_PERP_UNROLL" : "MACRO_TILE_WIDTH";
    std::string increment_string = load_to_lds_interwoven == 0 ? "++mu_perp_i" : "mu_perp_i += MACRO_TILE_WIDTH/MICRO_B_TILE_PERP_UNROLL";
    append_loop_var_bound_incr(ss, "mu_perp_i", bound_string, increment_string);
  }

  void append_b_load_for_pll(std::stringstream & ss){
    std::string bound_string = load_to_lds_interwoven == 0 ? "MICRO_B_TILE_PLL_UNROLL" : "UNROLL";
    std::string increment_string = load_to_lds_interwoven == 0 ? "++mu_pll_i" : "mu_pll_i += UNROLL/MICRO_B_TILE_PLL_UNROLL";
    append_loop_var_bound_incr(ss, "mu_pll_i", bound_string, increment_string);
  }

  void append_mn_factor_string(std::stringstream & ss){
    unsigned m_factor = use_edge_trick == 1 ? 1 : macro_tile_height;
    unsigned n_factor = use_edge_trick == 1 ? 1 : macro_tile_width;
    ss << R"(/* We define values which must be factors of m and n. Again, these have no influence on the running of the kernel */
/* If  use_edge_trick is true, these are just 1 (every m and n are permissible) otherwise they are macro-tile dimensions */
/* They are used by host code during checks for compatibility (for kernels with use_edge_trick false) of tile size with m and n */)" << 
    "\n" << "#define M_FACTOR " << m_factor << "\n" << "#define N_FACTOR " << n_factor;
  }
  
  void append_final_write_element(std::stringstream & ss, unsigned atomic_increment, unsigned with_beta_scaling, unsigned with_alpha_increment){
    
    ss << "\nindex = row_stride_c*(write_start_row + row) + col_stride_c*(write_start_col + col);\n";
    /* beta string */
    ss << (with_beta_scaling == 0 ? "" : "c[index] *= beta;\n");

    if (with_alpha_increment != 0){
      ss << "\n";
      if (atomic_increment == 0){
        ss << "c[index] += " + alpha_scaled + ";\n";
      }
      
      else{
        ss  
        << "ptr_to_c_elm = c + index;\n" 
        << "do {\n"
        << "previous_value = *ptr_to_c_elm;\n" 
        << "prevVal = as_" << infa << "(previous_value);\n"
        << "newVal = as_" << infa << "(" << alpha_scaled << " + previous_value);\n"
        << "} while (" << fati << "(( __global " << infa << "*)(ptr_to_c_elm), prevVal, newVal) != prevVal);";        
      }
    }
  }
  
  void append_for_loops_for_c_write_open(std::stringstream & ss){
    
    ss << "\n/* loops for writing to c */\n" << pragma_unroll_string;
    append_loop_var_bound_incr(ss, "row", 
    c_micro_tiles_interwoven == 0 ? "MICRO_TILE_HEIGHT" : "MACRO_TILE_HEIGHT", 
    c_micro_tiles_interwoven == 0 ? "++row" : "row += N_MICRO_TILES_VERTICALLY");
    ss << " {\n" << pragma_unroll_string;
    append_loop_var_bound_incr(ss, "col",
    c_micro_tiles_interwoven == 0 ? "MICRO_TILE_WIDTH" : "MACRO_TILE_WIDTH", 
    c_micro_tiles_interwoven == 0 ? "++col" : "col += N_MICRO_TILES_HORIZONTALLY");
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
((write_start_col + col >= MACRO_TILE_WIDTH*(n_groups_horizontally - 1)) && group_id_vertical   != (n_groups_vertically   - 1 )) ||
((write_start_row + row >= MACRO_TILE_HEIGHT*(n_groups_vertically - 1 )) && group_id_horizontal != (n_groups_horizontally - 1 )) ||
(
group_id_vertical == (n_groups_vertically-1)     && 
group_id_horizontal == (n_groups_horizontally-1) && 
write_start_col + col >= MACRO_TILE_WIDTH*(n_groups_horizontally - 1) && 
write_start_row + row >= MACRO_TILE_HEIGHT*(n_groups_vertically - 1)
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
    if (split_on_k == 0){
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
    ss << "\nunsigned k_remaining = " << effective_k <<  " % UNROLL;";
  }
  
  void append_worktime_increment_ab(std::stringstream & ss, unsigned final_unroll){
    if (final_unroll == 0){
      std::string n_jumps_string = split_on_k == 0 ? "UNROLL" : "G_UNROLL";
      ss << "a += col_stride_a*" << n_jumps_string << ";\nb += row_stride_b*" << n_jumps_string << ";\n";
    }
  }
  
  /* simple for loops. Could consider unrolling like Cobalt, but for the moment I use the optional pragma unroll */
  void append_load_ab_into_LDS_string(std::stringstream & ss, unsigned final_unroll, unsigned special_first_unroll){
    
    if (final_unroll != 0 && special_first_unroll != 0){
      throw std::runtime_error("From get_load_ab_into_LDS_string > It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out");
    }

    std::string a_value_to_get;
    std::string b_value_to_get;
    std::string a_comment;
    std::string b_comment;
    
    if (final_unroll == 1){
      a_value_to_get = "(a_offset_pll_unroll + mu_pll_i) < k_remaining ? a[mu_pll_i*col_stride_a + mu_perp_i*row_stride_a] : 0;";
      b_value_to_get = "(b_offset_pll_unroll + mu_pll_i) < k_remaining ? b[mu_pll_i*row_stride_b + mu_perp_i*col_stride_b] : 0;";
      a_comment =  "/* load final bit of data from a into LDS, less than a full unroll */";
      b_comment =  "/* load final bit of data from b into LDS, less than a full unroll */";
    }
    
    else if (special_first_unroll == 1){
      a_value_to_get = "(a_offset_pll_unroll + mu_pll_i) >= unroll_offset ? a[mu_pll_i*col_stride_a + mu_perp_i*row_stride_a] : 0;";
      b_value_to_get = "(b_offset_pll_unroll + mu_pll_i) >= unroll_offset ? b[mu_pll_i*row_stride_b + mu_perp_i*col_stride_b] : 0;";
      a_comment =  "/* load first bit of data from a into LDS, ignoring the prepended values (less than a full unroll)  */";
      b_comment =  "/* load first bit of data from b into LDS, ignoring the prepended values (less than a full unroll) */";
    }
    
    else{
      a_value_to_get = "a[mu_pll_i*col_stride_a + mu_perp_i*row_stride_a];";
      b_value_to_get = "b[mu_pll_i*row_stride_b + mu_perp_i*col_stride_b];";
      a_comment =  "/* load data from a into LDS */";
      b_comment =  "/* load data from b into LDS */";
    }
    
    ss << "\n" << a_comment << "\n"
    << pragma_unroll_string;
    append_a_load_for_perp(ss);
    ss << " {\n" 
    << pragma_unroll_string;
    append_a_load_for_pll(ss);
    ss << " {\n"
    << "localA[MACRO_TILE_HEIGHT_AND_PAD*(a_offset_pll_unroll + mu_pll_i) + (a_offset_perp_unroll + mu_perp_i)] = \n" 
    << a_value_to_get << "\n" 
    <<  "}\n" 
    <<  "}\n";


    ss 
    << "\n" << b_comment << "\n"
    << pragma_unroll_string;
    append_b_load_for_perp(ss);
    ss << " {\n" 
    << pragma_unroll_string;
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
    return c_micro_tiles_interwoven != 0 ? "1" : "MICRO_TILE_HEIGHT";
  }
  
  
  std::string get_c_work_item_horizontal_next(){
    return c_micro_tiles_interwoven != 0 ? "1" : "MICRO_TILE_WIDTH";
  }
      
  void append_relocate_lAlB_string(std::stringstream & ss){ //, unsigned final_unroll){
    ss << 
    "\n" << "lA = localA + micro_id_vertical*" << get_c_work_item_vertical_next() << ";" <<  
    "\n" << "lB = localB + micro_id_horizontal*" << get_c_work_item_horizontal_next() << ";" << "\n";
  }
  
  
  /* We previously had a variable unroll_the_math_section = False. */
  /* Experiments with unroll_the_math_section suggest that it's a bad idea. */
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
      throw std::runtime_error("From get_relocate_load_math_string : It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out");
    }

    append_load_ab_into_LDS_string(ss, final_unroll, special_first_unroll);
    ss << 
R"(
/* make sure all loads from LDS memory have completed */
barrier(CLK_LOCAL_MEM_FENCE); )";

    append_relocate_lAlB_string(ss);//, final_unroll);
    append_math_section(ss, final_unroll);
    ss << 
R"(
/* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
barrier(CLK_LOCAL_MEM_FENCE); )";
  }
    
  void append_final_unroll_string(std::stringstream & ss){
    
    if (split_on_k == 0){
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
    if (unroll_for_offset != 0){
      ss << "\n\n/* This is where the first unroll will be performed. Identical to what is in the main while, but with zero buffering.  */";
      if (split_on_k == 0){
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
    ss << pragma_unroll_string << "for (unsigned row = 0; row < MICRO_TILE_HEIGHT; ++row){\n" << pragma_unroll_string << "for (unsigned col = 0; col < MICRO_TILE_WIDTH; ++col){" << "\nrC[row][col] += rA[row]*rB[col]; // rA[row]*rB[col];  //mad(rA[row],rB[col],rC[row][col]);\n}\n}\n";
  }
  
  /* This returns the section which makes the within work-group adjust to a, b to
   * put a work item in the correct position to load its first element from global
   * if the load tiles are interlaced (ala cobalt), this final offset is just 1
   * row or column. If the tiles are not interlaced, this final offset is the width
   * or height of the load tile. */    
  void append_micro_offset_string(std::stringstream & ss){
    
    std::string str_a_n_pll(""); 
    std::string str_a_n_perp(""); 
    std::string str_b_n_pll("");
    std::string str_b_n_perp("");
    
    if (load_to_lds_interwoven == 0){
      str_a_n_pll = "MICRO_A_TILE_PLL_UNROLL *";
      str_a_n_perp = "MICRO_A_TILE_PERP_UNROLL *";
      str_b_n_pll = "MICRO_B_TILE_PLL_UNROLL *";
      str_b_n_perp = "MICRO_B_TILE_PERP_UNROLL *";
    }
    
    ss << "\n/* make the micro adjustments (A) for the thread, getting ready to load */\n";
    ss << "const unsigned a_offset_pll_unroll = " << str_a_n_pll << " pll_unroll_a_load_id;\n";
    ss <<  "const unsigned a_offset_perp_unroll = " << str_a_n_perp <<  " perp_unroll_a_load_id;\n";
    ss << "a += col_stride_a * a_offset_pll_unroll;\na += row_stride_a * a_offset_perp_unroll;\n\n";

    ss << "/* make the micro adjustments (B) for the thread, getting ready to load */\n";
    ss << "const unsigned b_offset_pll_unroll = " << str_b_n_pll << " pll_unroll_b_load_id;\n";
    ss << "const unsigned b_offset_perp_unroll = " << str_b_n_perp << " perp_unroll_b_load_id;\n";
    ss << "b += row_stride_b * b_offset_pll_unroll;\nb += col_stride_b * b_offset_perp_unroll;\n";
  }
  
  void append_load_load_string(std::stringstream & ss){
    ss << "\n" << pragma_unroll_string << "for (unsigned i = 0; i < MICRO_TILE_HEIGHT; ++i){\nrA[i] = lA[" << strided_i_vertical << "];\n}\n";
    ss << "lA += MACRO_TILE_HEIGHT_AND_PAD;\n\n" << pragma_unroll_string;
    ss << "for (unsigned i = 0; i < MICRO_TILE_WIDTH; ++i){\nrB[i] = lB[" << strided_i_horizontal << "];\n}\n";
    ss << "lB += MACRO_TILE_WIDTH_AND_PAD;";
  }

  void append_localA_localB_decl_string(std::stringstream & ss){
    ss << "__local TFLOAT localA[N_ELEMENTS_IN_PADDED_A_UNROLL];\n__local TFLOAT localB[N_ELEMENTS_IN_PADDED_B_UNROLL];\n";
  }
  
  
  void append_preshift_defns(std::stringstream & ss){
    
    if (use_edge_trick != 0){
      ss << "\nconst unsigned preshift_bottommost_tile_height = 1 + (m - 1) % MACRO_TILE_HEIGHT; // 1 ... MACRO_TILE_HEIGHT\n";
      ss << "const unsigned preshift_rightmost_tile_width = 1 + (n - 1) % MACRO_TILE_WIDTH; // 1 ... MACRO_TILE_WIDTH\n";
    }
  }
  
  void append_special_case_edge_trick_string(std::stringstream & ss){
    ss << "\n";
    if (use_edge_trick != 0){
      ss <<
R"(/* Special case of the tile being on far right : pull the tile to the left just enough so that it doesn't overflow C */
if (group_id_horizontal == n_groups_horizontally - 1){
macro_tile_start_col_in_c -= (MACRO_TILE_WIDTH - preshift_rightmost_tile_width);
}
  
/* Special case of the tile being on the bottom : pull the tile up just enough so that it doesn't overflow C */    
if (group_id_vertical == n_groups_vertically - 1){
macro_tile_start_row_in_c -= (MACRO_TILE_HEIGHT - preshift_bottommost_tile_height);
}
)";

    }
  }

  void append_group_allocation_defn_string(std::stringstream & ss){
    ss << "#define GROUP_ALLOCATION " << group_allocation;
    if (group_allocation == 3){
      ss << "/* this variable is declared because we have GROUP_ALLOCATION type 3. */\n";
      ss << "/* It should define how many workgroups we expect to have active simulantaneuosly. */\n";
      ss << "#define N_TARGET_ACTIVE_WORKGROUPS " << n_target_active_workgroups;
    }
  }
  
  void append_ngroups_grid_string(std::stringstream & ss){
    ss << "\n/* the number of work groups vertically and horizontally. */\n/* note that this ignores n_work_items_per_c_elm, so that only one workgroup per c cell is considered */ ";
  
    if (use_edge_trick == 1){
      ss << "\nconst unsigned n_groups_vertically = m / MACRO_TILE_HEIGHT + (preshift_bottommost_tile_height != MACRO_TILE_HEIGHT);";
      ss << "\nconst unsigned n_groups_horizontally = n / MACRO_TILE_WIDTH + (preshift_rightmost_tile_width != MACRO_TILE_WIDTH);\n";
    }
    
    else{
      ss << "\nconst unsigned n_groups_vertically = m / MACRO_TILE_HEIGHT;";
      ss << "\nconst unsigned n_groups_horizontally = n / MACRO_TILE_WIDTH;\n";
    }
  }
  
  
  void append_final_write_all(std::stringstream & ss){
    
    if (use_edge_trick == 0){
      ss << "\n";
      append_final_write_loops_no_check(ss);
    }
    
    else{
      ss << 
R"(
/* the case where this is not an edge tile : will write to all cells */
if ((group_id_horizontal != n_groups_horizontally - 1 || preshift_rightmost_tile_width == MACRO_TILE_WIDTH) 
&& (group_id_vertical != n_groups_vertically - 1 || preshift_bottommost_tile_height == MACRO_TILE_HEIGHT)){
)";
      append_final_write_loops_no_check(ss);
      ss << "\n}\n\nelse{";
      append_final_write_loops_with_check(ss);
      ss << "\n}";
    }
  }
  
  void append_split_on_k_ab_offset_adjustment_string(std::stringstream & ss){
    ss << "\n";
    if (split_on_k != 0){
      ss <<
R"(
/* a,b are pointing to the top left of the region required by the macro tile, but this work group  */
/* might not process the whole of a and b. We now turn 90 and shift pointers a,b to the start for this wg */
a += UNROLL*group_id_z*col_stride_a;
b += UNROLL*group_id_z*row_stride_b;
)";
    }
  }
  
  void append_k_unroll_offset_initial_string(std::stringstream & ss){
    if (unroll_for_offset != 0){
      ss <<
R"(
/* this additional offset of a and b appears because UNROLL_FOR_OFFSET is 1 */
unsigned unroll_offset = (3*group_id_vertical + 11*group_id_vertical)%UNROLL;
unsigned k_plus_offset = k + unroll_offset;
a -= unroll_offset*col_stride_a;
b -= unroll_offset*row_stride_b;
)";
    }
  }
  
  void append_split_on_k_defns_string(std::stringstream & ss){
    if (split_on_k != 0){
      ss << 
R"(
/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL )" << n_work_items_per_c_elm*unroll << " // N_WORK_ITEMS_PER_C_ELM*UNROLL";
    }
  }
  
  void append_group_id_defns(std::stringstream & ss){
    if (split_on_k == 0){
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
  
  void append_stride_defn(std::stringstream & ss, char LETTER, char letter, std::string ldx,  unsigned transposed){
    
    unsigned transposed_xor_is_col_major = (transposed + is_col_major) % 2;
    
    ss << "\n/* To move from " << LETTER << "[row][col] to " << LETTER << "[row+1][col], how much should the pointer increment? As we have " << LETTER << "_TRANSPOSED = " << transposed << " and IS_COL_MAJOR = " << is_col_major << ", this is */\nconst unsigned row_stride_" << letter << " = " << (transposed_xor_is_col_major == 1 ? "1" : ldx) << ";\n";
    
    ss << "/* To move from " << LETTER << "[row][col] to " << LETTER << "[row][col+1], how much should the pointer increment? As we have " << LETTER << "_TRANSPOSED = " << transposed << " and IS_COL_MAJOR = " << is_col_major << ", this is */\nconst unsigned col_stride_" << letter << " = " << (transposed_xor_is_col_major == 1 ? ldx : "1") << ";\n";
    
  }
  
  void append_stride_defns(std::stringstream & ss){
    ss << "\n/*a performance note : moving these (row_stride_x, col_stride_x) definitions to precompiler does not improve memory use or speed. */\n";
    append_stride_defn(ss, 'A', 'a', "lda", a_transposed);
    append_stride_defn(ss, 'B', 'b', "ldb", b_transposed);
    append_stride_defn(ss, 'C', 'c', "ldc", c_transposed);
  }
  
  void append_rA_rB_decl_string(std::stringstream & ss){
    ss << "TFLOAT rA[MICRO_TILE_HEIGHT];\nTFLOAT rB[MICRO_TILE_WIDTH];\n";
  }
  
  void append_n_unrolls_remaining_string(std::stringstream & ss){
    std::string k_effective_mod_G_UNROLL = effective_k + " % G_UNROLL";
    std::string k_effective_div_G_UNROLL = effective_k + " / G_UNROLL";
    std::string k_effective_div_UNROLL = effective_k + " / UNROLL";
    
    if (split_on_k == 0){
      ss << "\nint n_unrolls_remaining = " << k_effective_div_UNROLL << ";";
    }
    
    else{
      ss << "\nconst int n_work_groups_with_1_more = (" << k_effective_mod_G_UNROLL << ") / UNROLL; //JN_NEW\n";
      ss << "\n// branching between work groups : some wgs have 1 more unroll to process.\nint n_unrolls_remaining = (" << k_effective_div_G_UNROLL;
      ss << ") +  (group_id_z < n_work_groups_with_1_more);";
       
    }
  }
  
  void append_what_string(std::stringstream & ss){
    ss << "A:" << a_transposed << "B" << b_transposed << "C" << c_transposed << "f" << float_size;
  }
  
  void append_kernel_name(std::stringstream & ss){
    ss << "TODO";
  }

  
  KernelStringSetStatus set_string(std::string & kernel_string){



    std::string set_dimensions_status("");
    if (work_item_load_a_pll_to_unroll == 0){
      set_dimensions_status += set_tile_dimensions(micro_a_tile_perp_unroll, micro_a_tile_pll_unroll, macro_tile_height, unroll, n_elements_of_a_to_load_per_workitem); 
    }
    else{
      set_dimensions_status += set_tile_dimensions(micro_a_tile_pll_unroll, micro_a_tile_perp_unroll, unroll, macro_tile_height, n_elements_of_a_to_load_per_workitem);
    }
    
    
    
    if (work_item_load_b_pll_to_unroll == 0){
      set_dimensions_status += set_tile_dimensions(micro_b_tile_perp_unroll, micro_b_tile_pll_unroll, macro_tile_width, unroll, n_elements_of_b_to_load_per_workitem); 
    }
    else{
      set_dimensions_status += set_tile_dimensions(micro_b_tile_pll_unroll, micro_b_tile_perp_unroll, unroll, macro_tile_width, n_elements_of_b_to_load_per_workitem);
    }
    
    if (set_dimensions_status != ""){
      return KernelStringSetStatus(set_dimensions_status);
    }

    n_micro_a_tiles_pll_unroll = unroll / micro_a_tile_pll_unroll;
    n_micro_b_tiles_pll_unroll = unroll / micro_b_tile_pll_unroll;
  
  
      
  
  std::stringstream set_status_stream;
  
  if (n_elements_in_a_unroll % n_workitems_per_workgroup != 0){
    set_status_stream << "this is not supported : n_workitems_per_workgroup (" << n_workitems_per_workgroup << ") is not a factor of n_elements_in_" <<  "a" << "_unroll (" << n_elements_in_a_unroll << "). Consider rounding unroll up. \n";
  }

  if (n_elements_in_b_unroll % n_workitems_per_workgroup != 0){
    set_status_stream << "this is not supported : n_workitems_per_workgroup (" << n_workitems_per_workgroup << ") is not a factor of n_elements_in_" <<  "b" << "_unroll (" << n_elements_in_b_unroll << "). Consider rounding unroll up. \n";
  }
  
  std::string set_status_stream_string = set_status_stream.str();
  if (set_status_stream_string != ""){
    return KernelStringSetStatus(set_status_stream_string);
  }
  
  
    
    
  
  std::stringstream ss;
    
  
  
  ss << 
R"(/* ***********************************************
* These parameters define WHAT this kernel does *
* *********************************************** */
)";

  ss << "#define IS_COL_MAJOR " << is_col_major << "\n";
  ss << "#define A_TRANSPOSED " << a_transposed << "\n";
  ss << "#define B_TRANSPOSED " << b_transposed <<  "\n";
  ss << "#define C_TRANSPOSED " << c_transposed <<  "\n";
  ss << "#define TFLOAT  "  << t_float << "\n";
  
  ss << 
R"(/* certain kernels can only do one or the other of the terms in c <- alpha a*b + beta c. */
/* TODO : if DOES_ALPHA_C_INC is 0, then alpha should not appear as a kernel parameter */
)";
  ss << "#define DOES_ALPHA_C_INC " << does_alpha_c_inc << "\n";
  ss << "#define DOES_BETA_A_B_INC 1" << "\n";
  
  ss << 
R"(
/* TODO : remove final barrier, not nec. Check performance is not mysteriously hit! */
/* TODO : beta = 1 optimisation */
/* TODO : investigate mad. When should one use this instead of standard overloads, += and * ? 


/* ****************************************
 * These parameters define HOW it does it *
 * *****************************************/
/* Defines a tile shape. Each thread will process a tile of this shape from C (if N_WORK_ITEMS_PER_C_ELM > 1 the processing is shared with threads in other WGs)  */
)";
  
  
  
  ss << "#define MICRO_TILE_WIDTH " << micro_tile_width<< "\n";
  ss << "#define MICRO_TILE_HEIGHT " << micro_tile_height << "\n";
  ss << "/* Area of C which a workgroup will process. Recall that a workgroup is made of several threads which share LDS memory */\n";
  ss << "#define MACRO_TILE_WIDTH " << macro_tile_width << "\n";
  ss << "#define MACRO_TILE_HEIGHT " << macro_tile_height << "\n";
  ss << "/* How much a workgroup load (global -> LDS) in the k-direction at each iteration of the outer-most loop */\n";
  ss << "#define UNROLL " << unroll  << "\n";
  ss << "/* padding in LDS to avoid bank conflicts*/\n";
  ss << "#define PAD " << pad << "\n";
  ss << "/* whether or not this kernel uses the edge trick (see documentation : (TODO, currently internal AMD document)) */\n";
  ss << "/* this precompiler defn has no direct influence on the running the kernel, implementation already done in make_kernel.py */\n";
  ss << "#define EDGETRICK " << use_edge_trick << "\n";
  ss << "/* the number of work items working on the same c element. if this is 1, there will be just one thread doing all k multiply-adds, */\n";
  ss << "/* otherwise if it is greater than 1, each thread will be computing ~ k / N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added at the end */ \n";
  ss << "#define N_WORK_ITEMS_PER_C_ELM " << n_work_items_per_c_elm << "\n";
  ss << "/* whether or not to shimmy the starting k, in an attempt to avoid cache line overuse for cases where lda/ldb are powers of 2 */\n";
  ss << "/* if 0, no shimmying. if 1, instead of starting at k = 0 workgroups start at some negative offset dependent on work group id */\n";
  ss << "/* in the same way as the final unroll populates LDS with zeros in k mod UNROLL != 0, the initial negative indices here populate with 0 */\n";
  ss << "#define UNROLL_FOR_OFFSET " << unroll_for_offset << "\n";
  
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
/* Note : if the load tiles are long in the direction of unroll, the destination tile in LDS is NOT contiguous  */
/* We include these parameters here as pre-processor variables, but the loading micro-tile shapes are set in make_kernel.py */
)";


  ss << "#define WORK_ITEM_LOAD_A_PLL_TO_UNROLL " << work_item_load_a_pll_to_unroll << "\n"; 
  ss << "#define WORK_ITEM_LOAD_B_PLL_TO_UNROLL " << work_item_load_b_pll_to_unroll << "\n";
  ss << "/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if the load tiles are truly contiguous tiles of A/B (0) */\n";
  ss << "/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */\n";
  ss << "#define LOAD_TO_LDS_INTERWOVEN " << load_to_lds_interwoven << "\n";
  ss << "/* Whether the micro tile being processed by a compute item is interwoven with other micro tiles (ala Cobalt, (1)) or if the micro tiles are contiguous in C */\n";
  ss << "/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */\n";
  ss << "#define C_MICRO_TILES_INTERWOVEN " << c_micro_tiles_interwoven << "\n";
  ss << "/* Whether to use the unroll pragma to encourage the compiler to unroll certain loops */\n";
  ss << "/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */\n";
  ss << "#define PRAGMA_UNROLL_FORLOOPS " << unroll_pragma << "\n";
  ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into registers, as compared to doing the math. */\n";
  ss << "/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */\n";
  ss << "#define N_PREFETCH_FOR_REGISTER_LOAD " << 0 << "\n";
  ss << "/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into LDS, as compared to the unroll loop */\n";
  ss << "/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */\n";
  ss << "#define N_PREFETCH_FOR_LDS_LOAD " << 0 << "\n";

  ss << 
R"(



/* *****************************************************************************
 * The following are all implied by the preceding: these are NOT free parameters!
 * **************************************************************************** */
/*  <+_+>
 *           <+_+>
 *      <+_+>
 * TODO : rerereconsider assignment of workitems to load and math regions, there may be some sweet overlap spot where values automatically in registers for math (?) */
)";
  
  


  ss << "\n#define MACRO_TILE_AREA "<< macro_tile_area <<"  // MACRO_TILE_WIDTH*MACRO_TILE_HEIGHT\n";
  ss << "#define MICRO_TILE_AREA "<< micro_tile_area <<" // MICRO_TILE_WIDTH * MICRO_TILE_HEIGHT\n";
  ss << "#define N_WORK_ITEMS_PER_WORKGROUP  "<< n_workitems_per_workgroup <<" // MACRO_TILE_AREA / MICRO_TILE_AREA\n";
  ss << "#define MACRO_TILE_HEIGHT_AND_PAD "<< macro_tile_height_and_pad <<" // MACRO_TILE_HEIGHT + PAD\n";
  ss << "#define MACRO_TILE_WIDTH_AND_PAD "<< macro_tile_width_and_pad <<" // MACRO_TILE_WIDTH + PAD\n";
  ss << "#define N_ELEMENTS_IN_A_UNROLL "<< n_elements_in_a_unroll <<" // MACRO_TILE_HEIGHT * UNROLL\n";
  ss << "#define N_ELEMENTS_IN_B_UNROLL "<< n_elements_in_b_unroll <<" // MACRO_TILE_WIDTH * UNROLL\n";
  ss << "#define N_ELEMENTS_IN_PADDED_A_UNROLL "<< n_elements_in_padded_a_unroll <<" // MACRO_TILE_HEIGHT_AND_PAD * UNROLL\n";
  ss << "#define N_ELEMENTS_IN_PADDED_B_UNROLL "<< n_elements_in_padded_b_unroll <<" // MACRO_TILE_WIDTH_AND_PAD * UNROLL\n";
  ss << "#define N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM "<< n_elements_of_a_to_load_per_workitem <<" // N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP\n";
  ss << "#define N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM "<< n_elements_of_b_to_load_per_workitem <<" // N_ELEMENTS_IN_B_UNROLL / N_WORK_ITEMS_PER_WORKGROUP\n";
  ss << "#define N_MICRO_TILES_VERTICALLY "<< n_micro_tiles_vertically <<" // MACRO_TILE_HEIGHT / MICRO_TILE_HEIGHT\n";
  ss << "#define N_MICRO_TILES_HORIZONTALLY "<< n_micro_tiles_horizontally <<" // MACRO_TILE_WIDTH / MICRO_TILE_WIDTH\n";
  
  ss << "\n/* MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM */\n";
  ss << "/* The dimensions of a tile in A loaded by a work item.  */\n";
  ss << "#define MICRO_A_TILE_PLL_UNROLL " << micro_a_tile_pll_unroll << " // size of the loaded tile, pll to unroll\n";
  ss << "#define MICRO_A_TILE_PERP_UNROLL " << micro_a_tile_perp_unroll << "\n";
  ss << "#define N_MICRO_A_TILES_PLL_UNROLL " << n_micro_a_tiles_pll_unroll << " // UNROLL / MICRO_A_TILE_PLL_UNROLL\n""";
  
  ss << "\n/*  MICRO_B_TILE_PLL_UNROLL * MICRO_B_TILE_PERP_UNROLL = N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM */\n";
  ss << "/* The dimensions of a tile in B read by a work item */\n";
  ss << "#define MICRO_B_TILE_PLL_UNROLL " << micro_b_tile_pll_unroll << "\n";
  ss << "#define MICRO_B_TILE_PERP_UNROLL " << micro_b_tile_perp_unroll << "\n";
  ss << "#define N_MICRO_B_TILES_PLL_UNROLL " << n_micro_b_tiles_pll_unroll << " // UNROLL / MICRO_B_TILE_PLL_UNROLL\n";
  
  append_mn_factor_string(ss);
  append_split_on_k_defns_string(ss);
  append_super_column_width_defn(ss);
  
  ss << "\n\n\n__attribute__((reqd_work_group_size(" << n_workitems_per_workgroup << ",1, 1)))\n";
  ss << "__kernel void gpu_";
  append_kernel_name(ss);
  
  ss << 
R"((
__global TFLOAT       *          c,
__global const TFLOAT * restrict a,
__global const TFLOAT * restrict b,
const TFLOAT alpha,
const TFLOAT beta,
const unsigned lda,
const unsigned ldb,
const unsigned ldc,
const unsigned m,
const unsigned n,
const unsigned k,
const unsigned a_offset,
const unsigned b_offset,
const unsigned c_offset  
)
{

/* In OpenCL, host code does not have access to raw data pointers. */
/* Host code works with cl_mem objects, which encapsulate and hide raw points. */
/* For this reason, host code CANNOT simply increment pointers to data, */
/* as one can do with pointers for CPU gemm, or cublas gemm for that matter. */
a += a_offset;
b += b_offset;
c += c_offset;



)";

  append_stride_defns(ss);
  append_group_id_defns(ss);
  ss << "const unsigned local_id = get_local_id(0);\n";
  append_preshift_defns(ss);
  append_ngroups_grid_string(ss);
  append_group_allocation_string(ss);
    
  ss << 
R"(
unsigned macro_tile_start_row_in_c = group_id_vertical*MACRO_TILE_HEIGHT;
unsigned macro_tile_start_col_in_c = group_id_horizontal*MACRO_TILE_WIDTH;  

)";

  append_special_case_edge_trick_string(ss);

  ss << 
R"(
/* move to the top left corner of a (top left corner of b) of region required by the macro tile */
a += macro_tile_start_row_in_c*row_stride_a;   
b += macro_tile_start_col_in_c*col_stride_b;
)";

  append_split_on_k_ab_offset_adjustment_string(ss);
  append_k_unroll_offset_initial_string(ss);

  ss << 
R"(
/* Define which rows or columns of A, B this thread will load from global into LDS */
const unsigned pll_unroll_a_load_id = local_id % N_MICRO_A_TILES_PLL_UNROLL;
const unsigned perp_unroll_a_load_id = local_id / N_MICRO_A_TILES_PLL_UNROLL;
const unsigned pll_unroll_b_load_id = local_id % N_MICRO_B_TILES_PLL_UNROLL;
const unsigned perp_unroll_b_load_id = local_id / N_MICRO_B_TILES_PLL_UNROLL;
)";

  append_micro_offset_string(ss);
  
  ss << 
R"(

/* Define which part of the C macro-tile this thread will process */
const unsigned micro_id_vertical = local_id % N_MICRO_TILES_VERTICALLY;
const unsigned micro_id_horizontal = local_id / N_MICRO_TILES_VERTICALLY;
  
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
  append_k_remaining_string(ss);
  append_final_unroll_string(ss);
  
  ss << "const unsigned write_start_row = macro_tile_start_row_in_c + micro_id_vertical*" << get_c_work_item_vertical_next() << ";\n";
  ss << "const unsigned write_start_col = macro_tile_start_col_in_c + micro_id_horizontal*" << get_c_work_item_horizontal_next() << ";\n";  
  ss << "unsigned index;\n";
  
  append_split_on_k_vardecl_write_string(ss);
  append_final_write_all(ss);
  ss << "\n}\n";
  
  kernel_string = std::move(ss.str());
  
  return KernelStringSetStatus("");

}
  
};


void indentify(std::string & source){
  std::string newsource;
  newsource.reserve(source.length());
  std::string::size_type last_lend = source.find("\n", 0);  
  std::string::size_type next_lend = source.find("\n", last_lend + 1);
  std::string::size_type next_open = source.find("{", 0);
  std::string::size_type next_close = source.find("}", 0);  
  newsource.append(source, 0, last_lend);
  int indent_level = 0;

  while (std::string::npos != next_lend){

    if (next_open < last_lend){
      indent_level += 1;
      next_open = source.find("{", next_open + 1);
    }
    else if (next_close < next_lend){
      indent_level -= 1;
      next_close = source.find("}", next_close + 1);
    }
    
    else{
      newsource.append("\n");
      for (int i = 0; i < indent_level; ++i){
        newsource.append("  ");
      }
      newsource.append(source, last_lend + 1, next_lend - last_lend - 1);
      last_lend = next_lend;
      next_lend = source.find("\n", next_lend + 1);
    }
  }
  
  newsource += source.substr(last_lend);
  source.swap(newsource);
}


KernelStringSetStatus set_kernel_string(

  std::string & kernel_string,
  
  std::string kernelname, // = "",
  
  /* geometry parameters */
  unsigned float_size,
  unsigned a_transposed,
  unsigned b_transposed,
  unsigned c_transposed,
  unsigned is_col_major,
  /* to add m,n,k, lda, ldb, ldc */
  
  
  /* hyper parameters */
  unsigned micro_tile_width, 
  unsigned micro_tile_height, 
  unsigned macro_tile_width, 
  unsigned macro_tile_height, 
  unsigned unroll, 
  unsigned pad, 
  unsigned group_allocation, 
  unsigned work_item_load_a_pll_to_unroll, 
  unsigned work_item_load_b_pll_to_unroll, 
  unsigned unroll_pragma, 
  unsigned load_to_lds_interwoven, 
  unsigned c_micro_tiles_interwoven, 
  unsigned n_work_items_per_c_elm,
  unsigned unroll_for_offset,
  unsigned n_target_active_workgroups){
  
  KernelString kstring(
  kernelname,
  float_size,
  a_transposed,
  b_transposed,
  c_transposed,
  is_col_major,
  micro_tile_width, 
  micro_tile_height, 
  macro_tile_width, 
  macro_tile_height, 
  unroll, 
  pad, 
  group_allocation, 
  work_item_load_a_pll_to_unroll, 
  work_item_load_b_pll_to_unroll, 
  unroll_pragma, 
  load_to_lds_interwoven, 
  c_micro_tiles_interwoven, 
  n_work_items_per_c_elm,
  unroll_for_offset,
  n_target_active_workgroups);
  
  KernelStringSetStatus kernel_string_status = kstring.set_string(kernel_string);
  if (kernel_string_status.is_good()){
    indentify(kernel_string); /* make it prettier */
  }
  return kernel_string_status;
}


}
}
